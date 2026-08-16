#include "GigaLearnInfer/InferUnit.h"

#include <GigaLearnInfer/Models.h>
#include <GigaLearnInfer/ModelManager.h>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		ModelManager::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			useGPU ? torch::kCUDA : torch::kCPU,
			*this->models
		);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

GGL::InferUnit::~InferUnit() {
	if (models) {
		models->Free();
		delete models;
		models = nullptr;
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

int GGL::InferUnit::InferActionIndex(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActionIndices({ player }, { state }, deterministic, temperature)[0];
}

std::vector<int> GGL::InferUnit::BatchInferActionIndices(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(players[i], states[i]);
		if (curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
				"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
				"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
			);
		}
		allObs += curObs;

		allActionMasks += actionParser->GetActionMask(players[i], states[i]);
	}
	if (captureDiagnostics) {
		lastBatchSize = batchSize;
		lastObs = allObs;
		lastActionMasks = allActionMasks;
	}

	std::vector<int> results = {};

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tObs = torch::tensor(allObs).reshape({(int64_t)players.size(), obsSize});
		auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)players.size(), this->actionParser->GetActionAmount()});

		tObs = tObs.to(device);
		tActionMasks = tActionMasks.to(device);

		torch::Tensor tActions;
		if (captureDiagnostics) {
			torch::Tensor tProbs = ModelManager::InferPolicyProbsFromModels(*models, tObs, tActionMasks, temperature, false);
			lastActionProbs = TENSOR_TO_VEC<float>(tProbs.flatten());

			tActions = deterministic
				? tProbs.argmax(1)
				: torch::multinomial(tProbs, 1, true).flatten();
		} else {
			torch::Tensor tLogProbs;
			ModelManager::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);
		}

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);
		results.insert(results.end(), actionIndices.begin(), actionIndices.end());

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	std::vector<int> actionIndices = BatchInferActionIndices(players, states, deterministic, temperature);
	std::vector<RLGC::Action> results = {};
	results.reserve(actionIndices.size());

	for (int i = 0; i < (int)actionIndices.size(); i++)
		results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

	return results;
}