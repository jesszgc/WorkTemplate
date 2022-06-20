#include"Comm.h"
#include"3rdwrap/cuda/CudaService.h"
YzData::~YzData()
{
	if (im_bgrchar_gpu != nullptr) {
		cudaFree(im_bgrchar_gpu);
		im_bgrchar_gpu = nullptr;//?
	}
}

int YzData::init(int width, int height, int channels, int maxbatchsize, int gpudevice)
{
	gpu = gpudevice;
	cudaSetDevice(gpu);
	w_gpu = width;
	h_gpu = height;
	batchsize = maxbatchsize;


	im_bgrchar_gpu = safeCudaMalloc(sizeof(unsigned char) * channels * height * width * batchsize);
	if (im_bgrchar_gpu == nullptr)
	{
		return -1;
	}
	im_bgrchar_cpus.clear();
	w_cpus.clear();
	h_cpus.clear();
	GpuOrCpu = 0;

	return 0;
}

int YzData::PutData(std::vector<YzInData>& data)
{
	if (data.size() < 1)
	{
		return -1;
	}
	if (data.size() > batchsize)
	{
		std::cerr << "putdata pic num must less than yzdata batchsize" << std::endl;
		return -1;
	}
	int sizediffer = -1;
	for (int i = 0; i < data.size(); ++i)
	{
		if (data[i].height != h_gpu || data[i].width != w_gpu)
		{
			sizediffer = 1;
		}
	}

	if (sizediffer > 0) {
		im_bgrchar_cpus.clear();
		w_cpus.clear();
		h_cpus.clear();
		for (int i = 0; i < data.size(); ++i)
		{
			switch (data[i].datatype)
			{
			case InDataType::CPU_BGR:
			{
				im_bgrchar_cpus.push_back(data[i].img);
				w_cpus.push_back(data[i].width);
				h_cpus.push_back(data[i].height);
				break;
			}
			default:
			{
				return -1;
				break;
			}
			}
		}
		GpuOrCpu = -1;
	}
	else
	{
		for (int i = 0; i < data.size(); ++i)
		{
			switch (data[i].datatype)
			{
			case InDataType::CPU_BGR:
			{
				cudaMemcpy(((unsigned char*)im_bgrchar_gpu + i * w_gpu * h_gpu * 3), data[i].img, w_gpu * h_gpu * 3 * sizeof(char), cudaMemcpyHostToDevice);
				break;
			}
			case InDataType::GPU_CUDA_BGR:
			{
				cudaMemcpy(((unsigned char*)im_bgrchar_gpu + i * w_gpu * h_gpu * 3), data[i].img, w_gpu * h_gpu * 3 * sizeof(char), cudaMemcpyDeviceToDevice);
				break;
			}
			case InDataType::GPU_GpuMat:
			{
				int height = data[i].height;
				int width = data[i].width;
				int channels = data[i].channles;
				int step = data[i].step;
				unsigned char* pIn = (unsigned char*)im_bgrchar_gpu + i * sizeof(unsigned char) * height * width * channels;
				cudaMemcpy2D(pIn, width * channels, data[i].img, step, width * channels, height, cudaMemcpyDeviceToDevice);

				break;
			}
			case InDataType::GPU_GpuMat_BGRA:
			{

				unsigned char* pIn = (unsigned char*)im_bgrchar_gpu + i * sizeof(unsigned char) * h_gpu * w_gpu * 3;
				cudahelper::CVGpuMatBGRA_BGR((unsigned char*)data[i].img, data[i].width, data[i].height, data[i].step, pIn);
				break;
			}
			default:
			{
				return -1;
				break;
			}
			}
		}
		GpuOrCpu = 1;
	}
	picnum = data.size();
	return 0;

}

