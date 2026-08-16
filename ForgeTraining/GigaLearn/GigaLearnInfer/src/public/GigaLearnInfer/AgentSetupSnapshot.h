#pragma once

#include <GigaLearnInfer/Agent.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace RocketForge
{
	/** Serializable subset of GGL::PartialModelConfig used by Agent setup metadata. **/
	struct PartialModelConfigSnapshot
	{
		std::vector<int> layerSizes;
		GGL::ModelActivationType activationType = GGL::ModelActivationType::RELU;
		GGL::ModelOptimType optimType = GGL::ModelOptimType::ADAM;
		bool addLayerNorm = true;
		bool addOutputLayer = true;
	};

	/** Public Agent configuration that affects training and inference compatibility. **/
	struct AgentSetupSnapshot
	{
		std::string agentName;
		std::string obsBuilderName;
		std::string actionParserName;
		RocketSim::GameMode gameMode = RocketSim::GameMode::SOCCAR;
		RocketSim::CarConfig carConfig = RocketSim::CAR_CONFIG_OCTANE;
		int tickSkip = 8;
		int actionDelay = 7;
		int maxTeamSize = 1;
		float ballPredTickRate = 0.f;
		bool useBallPrediction = false;
		bool addLayerNorm = true;
		GGL::ModelOptimType optimType = GGL::ModelOptimType::ADAM;
		GGL::ModelActivationType activationType = GGL::ModelActivationType::RELU;
		PartialModelConfigSnapshot sharedHead;
		PartialModelConfigSnapshot policy;
		PartialModelConfigSnapshot critic;
	};

	/** One saved-vs-current Agent setup change. **/
	struct AgentSetupDiff
	{
		std::string field;
		std::string savedValue;
		std::string currentValue;
	};

	/** Captures public Agent configuration, excluding runtime modelPath. **/
	AgentSetupSnapshot CaptureAgentSetupSnapshot(const Agent& agent);

	/** Applies a previously captured setup to an Agent instance before stage construction. **/
	void ApplyAgentSetupSnapshot(Agent& agent, const AgentSetupSnapshot& snapshot);

	/** Returns display-ready saved -> current Agent setup differences. **/
	std::vector<AgentSetupDiff> DiffAgentSetupSnapshots(
		const AgentSetupSnapshot& saved,
		const AgentSetupSnapshot& current);

	/** Returns true when two Agent setup snapshots have no tracked differences. **/
	bool SameAgentSetupSnapshot(const AgentSetupSnapshot& a, const AgentSetupSnapshot& b);

	nlohmann::json AgentSetupSnapshotToJson(const AgentSetupSnapshot& snapshot);
	std::optional<AgentSetupSnapshot> AgentSetupSnapshotFromJson(const nlohmann::json& j);
}
