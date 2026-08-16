#pragma once

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include "ModelConfig.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace GGL {
	class ModelSet;

	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		ModelSet* models;
		bool useGPU;
		/** Stores obs, masks, and policy probabilities for optional RLBot analysis. **/
		bool captureDiagnostics = false;
		std::vector<float> lastObs;
		std::vector<uint8_t> lastActionMasks;
		std::vector<float> lastActionProbs;
		int lastBatchSize = 0;

		/** The owner resets obsBuilder before the first inference of each episode. **/
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU);
		~InferUnit();

		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);
		int InferActionIndex(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<int> BatchInferActionIndices(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

	};
}