#include "decode_yolov8_seg.h"

__global__ void decode_yolov8_seg_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh, float* src, int srcWidth, 
												int srcHeight, int srcArea, float* dst, int dstWidth, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; 
	int dy = blockDim.y * blockIdx.y + threadIdx.y; 
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;
	float* class_confidence = pitem + 4;
	// 第一个类别的概率 confidence, label = 0 
	float confidence = *class_confidence++; 
	int label = 0;
	// Get Max {80 class probability} and chass
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
	// 原子操作记录有效框数量：
	// dst的第一个元素用于存储有效框的数量
	int index = atomicAdd(dst + dy * dstArea, 1);
	if (index >= topK)
	{
		return;
	}

	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;

	float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left;
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;
	// 116 = 84+32； 39 = 7+32 
	memcpy(pout_item, pitem + num_class, 32 * sizeof(float));
}

void yolov8seg::decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, 
								int dstHeight)
{
	std::cout << "Debug decodeDevice Start:" << std::endl;
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 第一个标志位 记录有效框数量：
	int dstArea = 1 + dstWidth * dstHeight;
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
	decode_yolov8_seg_device_kernel <<< grid_size, block_size, 0, nullptr >>> (param.batch_size, param.num_class, param.topK, 
																				param.conf_thresh, src, srcWidth, srcHeight, srcArea, dst, 
																				dstWidth, dstArea);
	std::cout << "Debug decodeDevice Ends:" << std::endl;
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

void yolov8seg::transposeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, 
								int dstHeight)
{
	std::cout << "Debug transposeDevice Start:" << std::endl;
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE); 
	int dstArea = dstWidth * dstHeight;
	std::cout << "	srcWidth: " << srcWidth << std::endl;
	std::cout << "	srcHeight: " << srcHeight << std::endl;
	std::cout << "	srcArea: " << srcArea << std::endl;
	std::cout << "	dstWidth: " << dstWidth << std::endl;
	std::cout << "	dstHeight: " << dstHeight << std::endl;
	std::cout << "	dstArea: " << dstArea << std::endl;		
	transpose_device_kernel <<< grid_size, block_size, 0, nullptr >>> (param.batch_size, src, srcWidth, srcHeight, srcArea, dst, dstWidth, 
																		dstHeight, dstArea);
	std::cout << "Debug transposeDevice Ends:" << std::endl;
}

