#pragma once

#include <torch/torch.h>
#include "ModelConfig.h"

#include <cstdint>

namespace GGL {
	class Model;
	class ModelSet;

	/** Selects exact graph shapes or reusable power-of-two graph capacities. **/
	enum class CudaGraphBatchPolicy {
		Exact,
		PowerOfTwo
	};

	struct CudaGraphStats {
		uint64_t captures = 0;
		uint64_t replays = 0;
		uint64_t fallbacks = 0;
		uint64_t captureFailures = 0;
		uint64_t variantLimitFallbacks = 0;
		uint64_t cacheEntries = 0;
	};

	class RG_IMEXPORT ModelManager {
	public:
		static void MakeModels(
			bool makeCritic,
			int obsSize, int numActions,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels
		);

		static torch::Tensor InferCritic(torch::Tensor obs, ModelSet& models, bool useHalfPrecision);

		/**
		 * CUDA-graph inference reuses static output storage. Calls through either
		 * inference method below, and cache cleanup, must remain serialized on one
		 * host inference thread.
		 **/
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			float temperature,
			bool halfPrec,
			float minActionProbability = 0,
			bool useCudaGraph = false,
			CudaGraphBatchPolicy cudaGraphBatchPolicy = CudaGraphBatchPolicy::PowerOfTwo
		);
		static void InferActionsFromModels(
			ModelSet& models,
			torch::Tensor obs, torch::Tensor actionMasks,
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			float minActionProbability = 0,
			bool useCudaGraph = false,
			CudaGraphBatchPolicy cudaGraphBatchPolicy = CudaGraphBatchPolicy::PowerOfTwo
		);

		/** Frees any cached CUDA graphs captured from this model. Called automatically when a Model is destroyed. **/
		static void ReleaseCudaGraphs(const Model* model);

		static CudaGraphStats GetCudaGraphStats();
		static void ResetCudaGraphStats();
		static void ClearCudaGraphCache();
	};
} // namespace GGL
