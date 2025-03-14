#include"yolov8_seg.h"
#include"decode_yolov8_seg.h"

YOLOv8Seg::YOLOv8Seg(const utils::InitParameter& param) :yolo::YOLO(param)
{
    m_output_objects_device = nullptr;
    m_output_objects_width = 39; 
    m_output_src_width = 116;
    m_output_obj_area = 1 + param.topK * m_output_objects_width;
    m_output_seg_w = 160 * 160;
    m_output_seg_h = 32;
    int output_objects_size = param.batch_size * m_output_obj_area; 
    CHECK(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];

    m_mask160 = cv::Mat::zeros(1, 160 * 160, CV_32F);
    m_mask_eigen160 = Eigen::MatrixXf(1, 160 * 160);
    m_thresh_roi160 = cv::Rect(0, 0, 160, 160);
    m_thresh_roisrc = cv::Rect(0, 0, m_param.src_w, m_param.src_h);
    m_downsample_scale = 160.f / 640;
    m_mask_src = cv::Mat::zeros(m_param.src_h, m_param.src_w, CV_32F);
    m_img_canvas = cv::Mat::zeros(cv::Size(m_param.src_w, m_param.src_h), CV_8UC3);
}

YOLOv8Seg::~YOLOv8Seg()
{
    CHECK(cudaFree(m_output_objects_device));
    CHECK(cudaFree(m_output_src_device));
    CHECK(cudaFree(m_output_src_transpose_device));
    CHECK(cudaFree(m_output_seg_device));
    delete[] m_output_objects_host;
    delete[] m_output_seg_host;
    m_output_src_device = nullptr;
}

bool YOLOv8Seg::init(const std::vector<unsigned char>& trtFile)
{
    std::cout << "YOLOV8Seg::init Debug Start: " << std::endl;
    if (trtFile.empty())
    {
        return false;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime = 
        std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime == nullptr)
    {
        return false;
    }
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr)
    {
        return false;
    }
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return false;
    }
    if (m_param.dynamic_batch)
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }

	// 列出所有的绑定信息
	int numBindings = this->m_engine->getNbBindings();
	std::cout << "	Total number of bindings: " << numBindings << std::endl;

	for (int i = 0; i < numBindings; i++) {
		nvinfer1::Dims dims = this->m_context->getBindingDimensions(i);
		std::string name = this->m_engine->getBindingName(i);
		bool isInput = this->m_engine->bindingIsInput(i);
		
		std::cout << "	Binding " << i << ": " << name 
				<< " (Type: " << (isInput ? "Input" : "Output") << ")" << std::endl;
		
		std::cout << "	Dimensions: [";
		for (int j = 0; j < dims.nbDims; j++) {
			std::cout << dims.d[j];
			if (j < dims.nbDims - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
		
		// 获取数据类型
		nvinfer1::DataType dataType = this->m_engine->getBindingDataType(i);
		std::cout << "	Data Type: ";
		switch (dataType) {
			case nvinfer1::DataType::kFLOAT: std::cout << "FLOAT"; break;
			case nvinfer1::DataType::kHALF: std::cout << "HALF"; break;
			case nvinfer1::DataType::kINT8: std::cout << "INT8"; break;
			case nvinfer1::DataType::kINT32: std::cout << "INT32"; break;
			case nvinfer1::DataType::kBOOL: std::cout << "BOOL"; break;
			default: std::cout << "UNKNOWN";
		}
		std::cout << std::endl << std::endl;
	}

	// [1,116,8400]
    m_output_dims = this->m_context->getBindingDimensions(2);
    m_total_objects = m_output_dims.d[2];
    std::cout << "	m_total_objects: " << m_total_objects << std::endl;
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1; // 116 * 8400
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
	std::cout << "	m_output_area: " << m_output_area << std::endl;
	
	// [1,32,160,160]
    m_output_seg_dims = this->m_context->getBindingDimensions(1);
    assert(m_param.batch_size <= m_output_seg_dims.d[0]);
    m_output_seg_area = 1; // 32 * 160 * 160
    for (int i = 1; i < m_output_seg_dims.nbDims; i++)
    {
        if (m_output_seg_dims.d[i] != 0)
        {
            m_output_seg_area *= m_output_seg_dims.d[i];
        }
    }
	std::cout << "	m_output_seg_area(32*160*160): " << m_output_seg_area << std::endl;
    CHECK(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    CHECK(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
    CHECK(cudaMalloc(&m_output_seg_device, m_param.batch_size * m_output_seg_area * sizeof(float)));
    m_output_seg_host = new float[m_param.batch_size * m_output_seg_area];   

    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
	std::cout << "	m_param.src_h: " << m_param.src_h << "	m_param.src_w: " << m_param.src_w << std::endl;
	std::cout << "	m_param.dst_h: " << m_param.dst_h << "	m_param.dst_w: " << m_param.dst_w << std::endl;
	std::cout << "	a: " << a << " b: " << b << " scale: " << scale << std::endl;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 
												0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);
    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];
	std::cout << "	m_dst2src.v0: " << m_dst2src.v0 << " m_dst2src.v1: " << m_dst2src.v1 << " m_dst2src.v2: " << m_dst2src.v2 << std::endl;
	std::cout << "	m_dst2src.v3: " << m_dst2src.v3 << " m_dst2src.v4: " << m_dst2src.v4 << " m_dst2src.v5: " << m_dst2src.v5 << std::endl;
    return true;
}

void YOLOv8Seg::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h, m_input_resize_device, m_param.dst_w, m_param.dst_h, 
					114, m_dst2src);
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h, m_input_rgb_device, m_param.dst_w, 
					m_param.dst_h);
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h, m_input_norm_device, m_param.dst_w, 
				m_param.dst_h, m_param);
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h, m_input_hwc_device, m_param.dst_w, m_param.dst_h);
}

bool YOLOv8Seg::infer()
{
    float* bindings[] = {m_input_hwc_device, m_output_seg_device, m_output_src_device};
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void YOLOv8Seg::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    yolov8seg::transposeDevice(m_param, m_output_src_device, m_total_objects, m_output_src_width, m_total_objects * m_output_src_width, 
								m_output_src_transpose_device, m_output_src_width, m_total_objects);
    yolov8seg::decodeDevice(m_param, m_output_src_transpose_device, m_output_src_width, m_total_objects, m_output_area, 
							m_output_objects_device, m_output_objects_width, m_param.topK);
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_output_obj_area);
    CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * m_output_obj_area, 
						cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_output_seg_host, m_output_seg_device, m_param.batch_size * sizeof(float) * m_output_seg_area, 
						cudaMemcpyDeviceToHost));
}

void YOLOv8Seg::reset()
{
    CHECK(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * m_output_obj_area));
}

void YOLOv8Seg::showAndSave(const std::vector<std::string>& classNames, const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch)
{
    std::cout << "YOLOv8Seg::showAndSave Debug Start: " << std::endl;
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = {bbox_points[0]};
    int num_points[] = {4};
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
		// 检测框数量
        int num_boxes = std::min((int)(m_output_objects_host + bi * m_output_obj_area)[0], m_param.topK);
        std::cout << "	num_boxes: " << num_boxes << std::endl;

        float* p_seg = m_output_seg_host + bi * m_output_seg_area;
        // [32,160,160] --> [160*160,32] 转秩; 原型掩码矩阵
        Eigen::Map<Eigen::MatrixXf> img_seg_(p_seg, m_output_seg_w, m_output_seg_h);
        std::cout << "	m_output_seg_h: " << m_output_seg_h << "; m_output_seg_w: " << m_output_seg_w << std::endl;
        int m_output_obj_area_bi = bi * m_output_obj_area;
        std::cout << "	m_output_obj_area_bi: " << m_output_obj_area_bi << std::endl;
        m_img_canvas.setTo(cv::Scalar(0, 0, 0));
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + m_output_obj_area_bi + m_output_objects_width * i + 1;
            // 校验是否被NMS滤掉
            if (ptr[6])
            {
                int label = ptr[5];
                std::cout << "	label: " << label << std::endl;
                cv::Scalar color = utils::Colors::color80[label];
                // [32,1] 分割编码系数
                Eigen::Map<Eigen::MatrixXf> img_ojb_seg_(ptr + 7, m_output_seg_h, 1); 
                // [160*160,1] 计算物体的分割掩码 , 掩码概率矩阵
                m_mask_eigen160 = img_seg_ * img_ojb_seg_;
                cv::eigen2cv(m_mask_eigen160, m_mask160);
                // sigmoid 函数: 1/(1+e^(-x)) 表示掩码概率
                cv::exp(-m_mask160, m_mask160); 
                m_mask160 = 1.f / (1.f + m_mask160); 
                // [25600,1] --> [160,160]
                m_mask160 = m_mask160.reshape(1, 160);
                std::cout << "	m_mask160.shape: " << m_mask160.rows << " " << m_mask160.cols << std::endl;
				std::cout << "	left: " << ptr[0] << "; top: " << ptr[1] << "; right: " << ptr[2] << "; bottom: " << ptr[3] << std::endl; 
                // 计算检测框在掩码尺度上的坐标 从[640,640]的坐标 映射到 [160,160]的坐标
                // ptr[0:3] left top right bottom
                int x_lt_160 = std::round(ptr[0] * m_downsample_scale); 
                int y_lt_160 = std::round(ptr[1] * m_downsample_scale);
                int x_rb_160 = std::round(ptr[2] * m_downsample_scale);
                int y_rb_160 = std::round(ptr[3] * m_downsample_scale);
				std::cout << "	x_lt_160: " << x_lt_160 << "; y_lt_160: " << y_lt_160 << "; x_rb_160: " << x_rb_160 << "; y_rb_160: " << y_rb_160 << std::endl;
                // 创建掩码尺度上的感兴趣区域（ROI）
				// 使用 & 运算符计算掩码尺度矩形与 m_thresh_roi160 的交集, 保证后续掩码操作不会越界
                cv::Rect roi160 = cv::Rect(x_lt_160, y_lt_160, x_rb_160 - x_lt_160, y_rb_160 - y_lt_160) & m_thresh_roi160;
                std::cout << "	roi160.shape: " << roi160.height << " " << roi160.width << std::endl;
                if (roi160.width == 0 || roi160.height == 0)
                    continue;
                
                // 计算检测框在原始图像尺度上的坐标
                int x_lt_src = std::round(m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2);
                int y_lt_src = std::round(m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5);
                int x_rb_src = std::round(m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2);
                int y_rb_src = std::round(m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5);
				std::cout << "	x_lt_src: " << x_lt_src << "; y_lt_src: " << y_lt_src << "; x_rb_src: " << x_rb_src << "; y_rb_src: " << y_rb_src << std::endl;
                // 保证后续操作不会超出原图像尺寸范围
				cv::Rect roisrc = cv::Rect(x_lt_src, y_lt_src, x_rb_src - x_lt_src, y_rb_src - y_lt_src) & m_thresh_roisrc;
                std::cout << "	roisrc.shape: " << roisrc.height << " " << roisrc.width << std::endl;
                if (roisrc.width == 0 || roisrc.height == 0)
                    continue;      

                // for opencv >=4.7(faster)
                // cv::Mat mask_instance; 
                // cv::resize(cv::Mat(m_mask160, roi160), mask_instance, cv::Size(roisrc.width, roisrc.height), cv::INTER_LINEAR);
                // mask_instance = mask_instance > 0.5f;
                // cv::cvtColor(mask_instance, mask_instance, cv::COLOR_GRAY2BGR);
                // mask_instance.setTo(color, mask_instance);          
                // cv::addWeighted(mask_instance, 0.45, m_img_canvas(roisrc), 1.0, 0., m_img_canvas(roisrc));

                // for opencv >=3.2.0
                cv::Mat mask_instance;
				// 从 掩码概率矩阵 m_mask160 中提取 roi160 定义的区域
				// 使用插值方法 i调整大小到 (roisrc.width, roisrc.height) 尺寸
                cv::resize(cv::Mat(m_mask160, roi160), mask_instance, cv::Size(roisrc.width, roisrc.height), cv::INTER_LINEAR);
                // 将掩码二值化 - 大于 0.5 的值设为 1（物体），小于等于 0.5 的值设为 0（背景）
                mask_instance = mask_instance > 0.5f;
                cv::Mat mask_instance_bgr;
                // 将单通道掩码转换为 3 通道 BGR 格式，以便应用颜色
                cv::cvtColor(mask_instance, mask_instance_bgr, cv::COLOR_GRAY2BGR);
                // 将掩码中的物体区域（值为1的区域）设置为当前类别的颜色
                mask_instance_bgr.setTo(color, mask_instance);
                // 将彩色掩码与图像画布叠加 掩码区域权重为 0.45，检测框区域权重为 1.0，创建半透明效果
                cv::addWeighted(mask_instance_bgr, 0.45, m_img_canvas(roisrc), 1.0, 0., m_img_canvas(roisrc));

                // label's info
                cv::rectangle(imgsBatch[bi], roisrc, color, 2, cv::LINE_AA);
                cv::String det_info = m_param.class_names[label] + " " + cv::format("%.4f", ptr[4]);
                // 顺时针四个定点
                bbox_points[0][0] = cv::Point(x_lt_src, y_lt_src);
                bbox_points[0][1] = cv::Point(x_lt_src + det_info.size() * m_param.char_width, y_lt_src);
                bbox_points[0][2] = cv::Point(x_lt_src + det_info.size() * m_param.char_width, y_lt_src - m_param.det_info_render_width);
                bbox_points[0][3] = cv::Point(x_lt_src, y_lt_src - m_param.det_info_render_width);
                cv::fillPoly(imgsBatch[bi], bbox_point0, num_points, 1, color);
                cv::putText(imgsBatch[bi], det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, m_param.font_scale, 
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }
        if (m_param.is_show)
        {
            cv::imshow(m_param.winname, imgsBatch[bi] + m_img_canvas);
            cv::waitKey(cvDelayTime);
        }
        if (m_param.is_save)
        {
            cv::imwrite(m_param.save_path + utils::getTimeStamp() + ".jpg", imgsBatch[bi] + m_img_canvas);
        }
    }
	std::cout << "YOLOv8Seg::showAndSave Debug Ends! " << std::endl;
}