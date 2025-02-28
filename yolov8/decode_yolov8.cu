#include "decode_yolov8.h"

// 这个算法 由于并行执行的 随机性 可能会出现较好的候选框因为处理顺序而被舍弃
// 当累计处理的有效框达到topK时，后续的框即使class_confidence高也会被丢弃
// 最终保留的可能不是"最好的"topK个框，而是"最先处理的"topK个框
// 
__global__ void decode_yolov8_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh, float* src, int srcWidth, 
											int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;
	// 84 = x,y,w,h + 80 class probability
	float* class_confidence = pitem + 4;
	// 拿到 Max{80 class probability}, 对应的 class
	float confidence = *class_confidence++;
	int label = 0;
	for (int i = 1; i < num_class; ++i, ++class_confidence)
	{
		if (*class_confidence > confidence)
		{
			confidence = *class_confidence;
			label = i;
		}
	}
	if (confidence < conf_thresh)
	{
		return;
	}
	// 有大部分超过 conf_thresh 
	// printf("	confidence: %f\n", confidence);

	// dst的第一个元素用于存储有效框的数量
	// 原子操作记录有效框数量：
	int index = atomicAdd(dst + dy * dstArea, 1);
	if (index >= topK)
	{
		return;
	}
	// 坐标转换 [x,y,w,h] --> [left,top,right,bottom]
	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;
	// 框的 H,W 不固定
	// printf("	cx = %f, cy = %f, width = %f, height = %f\n", cx,cy,width,height);
	float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;
	// dst的第一个元素用于存储有效框的数量
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left;
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = confidence;
	*pout_item++ = label;
	// 候选框中是否存在物体的置信度(框内存在物体的概率)
	*pout_item++ = 1;
}

void yolov8::decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, 
							int dstHeight)
{
	std::cout << "Debug decodeDevice Start: " << std::endl; 
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 第一个标志位 记录有效框数量：
	int dstArea = 1 + dstWidth * dstHeight;
	// [N,8400,84]
	std::cout << "	param.batch_size: " << param.batch_size << std::endl;
	std::cout << "	param.num_class: " << param.num_class << std::endl;
	std::cout << "	param.topK: " << param.topK << std::endl;
	std::cout << "	param.conf_thresh: " << param.conf_thresh << std::endl;
	std::cout << "	srcWidth: " << srcWidth << std::endl;
	std::cout << "	srcHeight: " << srcHeight << std::endl;
	std::cout << "	srcArea: " << srcArea << std::endl;
	std::cout << "	dstWidth: " << dstWidth << std::endl;
	std::cout << "	dstHeight: " << dstHeight << std::endl;
	std::cout << "	dstArea: " << dstArea << std::endl;
	decode_yolov8_device_kernel <<< grid_size, block_size, 0, nullptr >>> (param.batch_size, param.num_class, param.topK, param.conf_thresh,
																			src, srcWidth, srcHeight, srcArea, dst, dstWidth, dstHeight, 
																			dstArea);
}


__global__ void transpose_device_kernel(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth,
										int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx >= dstHeight || dy >= batch_size)
	{
		return;
	}
	float* p_dst_row = dst + dy * dstArea + dx * dstWidth;
	float* p_src_col = src + dy * srcArea + dx;

	for (int i = 0; i < dstWidth; i++)
	{
		p_dst_row[i] = p_src_col[i * srcWidth];
	}
}

void yolov8::transposeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, 
								int dstHeight)
{
	std::cout << "Debug TransposeDevice Start: " << std::endl;
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = dstWidth * dstHeight;
	// [N,84,8400] --> [N,8400,84]
	std::cout << "	param.batch_size: " << param.batch_size << std::endl;
	std::cout << "	srcWidth: " << srcWidth << std::endl;
	std::cout << "	srcHeight: " << srcHeight << std::endl;
	std::cout << "	srcArea: " << srcArea << std::endl;
	std::cout << "	dstWidth: " << dstWidth << std::endl;
	std::cout << "	dstHeight: " << dstHeight << std::endl;
	std::cout << "	dstArea: " << dstArea << std::endl;
	transpose_device_kernel <<< grid_size, block_size, 0, nullptr >>> (param.batch_size, src, srcWidth, srcHeight, srcArea, dst, dstWidth,
																		dstHeight, dstArea);
	std::cout << "Debug TransposeDevice Ends;" << std::endl;
}


