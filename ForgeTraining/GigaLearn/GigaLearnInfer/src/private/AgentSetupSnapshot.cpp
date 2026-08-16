#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>

using nlohmann::json;

namespace RocketForge
{
	namespace
	{
		std::string FormatBool(bool value)
		{
			return value ? "true" : "false";
		}

		std::string FormatFloat(float value)
		{
			std::ostringstream out;
			out << std::setprecision(9) << std::defaultfloat << value;
			return out.str();
		}

		bool NearlyEqual(float a, float b)
		{
			return std::fabs(a - b) <= 0.0001f;
		}

		bool SameVec(const RocketSim::Vec& a, const RocketSim::Vec& b)
		{
			return NearlyEqual(a.x, b.x)
				&& NearlyEqual(a.y, b.y)
				&& NearlyEqual(a.z, b.z)
				&& NearlyEqual(a._w, b._w);
		}

		bool SameWheelPairConfig(const RocketSim::WheelPairConfig& a, const RocketSim::WheelPairConfig& b)
		{
			return NearlyEqual(a.wheelRadius, b.wheelRadius)
				&& NearlyEqual(a.suspensionRestLength, b.suspensionRestLength)
				&& SameVec(a.connectionPointOffset, b.connectionPointOffset);
		}

		bool SameCarConfig(const RocketSim::CarConfig& a, const RocketSim::CarConfig& b)
		{
			return SameVec(a.hitboxSize, b.hitboxSize)
				&& SameVec(a.hitboxPosOffset, b.hitboxPosOffset)
				&& SameWheelPairConfig(a.frontWheels, b.frontWheels)
				&& SameWheelPairConfig(a.backWheels, b.backWheels)
				&& a.threeWheels == b.threeWheels
				&& NearlyEqual(a.dodgeDeadzone, b.dodgeDeadzone);
		}

		std::string FormatLayers(const std::vector<int>& layerSizes)
		{
			if (layerSizes.empty())
				return "[]";

			std::ostringstream out;
			out << "[";
			for (size_t i = 0; i < layerSizes.size(); ++i) {
				if (i > 0)
					out << ", ";
				out << layerSizes[i];
			}
			out << "]";
			return out.str();
		}

		std::string FormatGameMode(RocketSim::GameMode mode)
		{
			switch (mode) {
			case RocketSim::GameMode::SOCCAR: return "soccar";
			case RocketSim::GameMode::HOOPS: return "hoops";
			case RocketSim::GameMode::HEATSEEKER: return "heatseeker";
			case RocketSim::GameMode::SNOWDAY: return "snowday";
			case RocketSim::GameMode::DROPSHOT: return "dropshot";
			case RocketSim::GameMode::THE_VOID: return "void";
			default: return std::to_string(static_cast<int>(mode));
			}
		}

		std::string FormatCarConfig(const RocketSim::CarConfig& config)
		{
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_OCTANE)) return "Octane";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_DOMINUS)) return "Dominus";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_PLANK)) return "Plank";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_BREAKOUT)) return "Breakout";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_HYBRID)) return "Hybrid";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_MERC)) return "Merc";
			if (SameCarConfig(config, RocketSim::CAR_CONFIG_PSYCLOPS)) return "Psyclops";
			return "Custom";
		}

		std::string FormatOptimType(GGL::ModelOptimType type)
		{
			switch (type) {
			case GGL::ModelOptimType::ADAM: return "ADAM";
			case GGL::ModelOptimType::ADAMW: return "ADAMW";
			case GGL::ModelOptimType::ADAGRAD: return "ADAGRAD";
			case GGL::ModelOptimType::RMSPROP: return "RMSPROP";
			case GGL::ModelOptimType::MAGSGD: return "MAGSGD";
			default: return std::to_string(static_cast<int>(type));
			}
		}

		std::string FormatActivationType(GGL::ModelActivationType type)
		{
			switch (type) {
			case GGL::ModelActivationType::RELU: return "RELU";
			case GGL::ModelActivationType::LEAKY_RELU: return "LEAKY_RELU";
			case GGL::ModelActivationType::SIGMOID: return "SIGMOID";
			case GGL::ModelActivationType::TANH: return "TANH";
			default: return std::to_string(static_cast<int>(type));
			}
		}

		json VecToJson(const RocketSim::Vec& vec)
		{
			return json{
				{ "x", vec.x },
				{ "y", vec.y },
				{ "z", vec.z },
				{ "w", vec._w },
			};
		}

		RocketSim::Vec VecFromJson(const json& j)
		{
			RocketSim::Vec vec = {};
			if (!j.is_object())
				return vec;

			vec.x = j.value("x", 0.f);
			vec.y = j.value("y", 0.f);
			vec.z = j.value("z", 0.f);
			vec._w = j.value("w", 0.f);
			return vec;
		}

		json WheelPairConfigToJson(const RocketSim::WheelPairConfig& config)
		{
			return json{
				{ "wheel_radius", config.wheelRadius },
				{ "suspension_rest_length", config.suspensionRestLength },
				{ "connection_point_offset", VecToJson(config.connectionPointOffset) },
			};
		}

		RocketSim::WheelPairConfig WheelPairConfigFromJson(const json& j)
		{
			RocketSim::WheelPairConfig config = {};
			if (!j.is_object())
				return config;

			config.wheelRadius = j.value("wheel_radius", 0.f);
			config.suspensionRestLength = j.value("suspension_rest_length", 0.f);
			config.connectionPointOffset = VecFromJson(j.value("connection_point_offset", json::object()));
			return config;
		}

		std::string NormalizePresetName(std::string text)
		{
			text.erase(
				std::remove_if(text.begin(), text.end(), [](unsigned char c) {
					return c == '_' || c == '-' || std::isspace(c);
				}),
				text.end());
			std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
				return static_cast<char>(std::tolower(c));
			});
			return text;
		}

		json CarConfigToJson(const RocketSim::CarConfig& config)
		{
			const std::string preset = FormatCarConfig(config);
			if (preset != "Custom")
				return preset;

			return json{
				{ "hitbox_size", VecToJson(config.hitboxSize) },
				{ "hitbox_pos_offset", VecToJson(config.hitboxPosOffset) },
				{ "front_wheels", WheelPairConfigToJson(config.frontWheels) },
				{ "back_wheels", WheelPairConfigToJson(config.backWheels) },
				{ "three_wheels", config.threeWheels },
				{ "dodge_deadzone", config.dodgeDeadzone },
			};
		}

		RocketSim::CarConfig CarConfigFromPresetName(const std::string& name)
		{
			const std::string normalized = NormalizePresetName(name);
			if (normalized == "octane") return RocketSim::CAR_CONFIG_OCTANE;
			if (normalized == "dominus") return RocketSim::CAR_CONFIG_DOMINUS;
			if (normalized == "plank" || normalized == "batmobile") return RocketSim::CAR_CONFIG_PLANK;
			if (normalized == "breakout") return RocketSim::CAR_CONFIG_BREAKOUT;
			if (normalized == "hybrid") return RocketSim::CAR_CONFIG_HYBRID;
			if (normalized == "merc") return RocketSim::CAR_CONFIG_MERC;
			if (normalized == "psyclops") return RocketSim::CAR_CONFIG_PSYCLOPS;
			return RocketSim::CAR_CONFIG_OCTANE;
		}

		RocketSim::CarConfig CarConfigFromJson(const json& j)
		{
			if (j.is_string())
				return CarConfigFromPresetName(j.get<std::string>());

			RocketSim::CarConfig config = RocketSim::CAR_CONFIG_OCTANE;
			if (!j.is_object())
				return config;

			config.hitboxSize = VecFromJson(j.value("hitbox_size", json::object()));
			config.hitboxPosOffset = VecFromJson(j.value("hitbox_pos_offset", json::object()));
			config.frontWheels = WheelPairConfigFromJson(j.value("front_wheels", json::object()));
			config.backWheels = WheelPairConfigFromJson(j.value("back_wheels", json::object()));
			config.threeWheels = j.value("three_wheels", false);
			config.dodgeDeadzone = j.value("dodge_deadzone", 0.5f);
			return config;
		}

		PartialModelConfigSnapshot CapturePartialModelConfig(const GGL::PartialModelConfig& config)
		{
			PartialModelConfigSnapshot snapshot = {};
			snapshot.layerSizes = config.layerSizes;
			snapshot.activationType = config.activationType;
			snapshot.optimType = config.optimType;
			snapshot.addLayerNorm = config.addLayerNorm;
			snapshot.addOutputLayer = config.addOutputLayer;
			return snapshot;
		}

		GGL::PartialModelConfig ToPartialModelConfig(const PartialModelConfigSnapshot& snapshot)
		{
			GGL::PartialModelConfig config = {};
			config.layerSizes = snapshot.layerSizes;
			config.activationType = snapshot.activationType;
			config.optimType = snapshot.optimType;
			config.addLayerNorm = snapshot.addLayerNorm;
			config.addOutputLayer = snapshot.addOutputLayer;
			return config;
		}

		json PartialModelConfigToJson(const PartialModelConfigSnapshot& config)
		{
			return json{
				{ "layer_sizes", config.layerSizes },
				{ "activation_type", static_cast<int>(config.activationType) },
				{ "optim_type", static_cast<int>(config.optimType) },
				{ "add_layer_norm", config.addLayerNorm },
				{ "add_output_layer", config.addOutputLayer },
			};
		}

		PartialModelConfigSnapshot PartialModelConfigFromJson(const json& j)
		{
			PartialModelConfigSnapshot config = {};
			if (!j.is_object())
				return config;

			if (j.contains("layer_sizes") && j["layer_sizes"].is_array()) {
				for (const json& layer : j["layer_sizes"]) {
					if (layer.is_number_integer())
						config.layerSizes.push_back(layer.get<int>());
				}
			}
			config.activationType = static_cast<GGL::ModelActivationType>(j.value("activation_type", 0));
			config.optimType = static_cast<GGL::ModelOptimType>(j.value("optim_type", 0));
			config.addLayerNorm = j.value("add_layer_norm", true);
			config.addOutputLayer = j.value("add_output_layer", true);
			return config;
		}

		void AddDiff(
			std::vector<AgentSetupDiff>& diffs,
			const std::string& field,
			const std::string& savedValue,
			const std::string& currentValue)
		{
			if (savedValue == currentValue)
				return;

			diffs.push_back({ field, savedValue, currentValue });
		}

		void AddPartialModelConfigDiffs(
			std::vector<AgentSetupDiff>& diffs,
			const std::string& prefix,
			const PartialModelConfigSnapshot& saved,
			const PartialModelConfigSnapshot& current)
		{
			AddDiff(diffs, prefix + " Layers", FormatLayers(saved.layerSizes), FormatLayers(current.layerSizes));
			AddDiff(diffs, prefix + " Activation", FormatActivationType(saved.activationType), FormatActivationType(current.activationType));
			AddDiff(diffs, prefix + " Optimizer", FormatOptimType(saved.optimType), FormatOptimType(current.optimType));
			AddDiff(diffs, prefix + " Layer Norm", FormatBool(saved.addLayerNorm), FormatBool(current.addLayerNorm));
			AddDiff(diffs, prefix + " Output Layer", FormatBool(saved.addOutputLayer), FormatBool(current.addOutputLayer));
		}
	}

	AgentSetupSnapshot CaptureAgentSetupSnapshot(const Agent& agent)
	{
		AgentSetupSnapshot snapshot = {};
		snapshot.agentName = agent.agentName;
		snapshot.obsBuilderName = agent.obsBuilderName;
		snapshot.actionParserName = agent.actionParserName;
		snapshot.gameMode = agent.gameMode;
		snapshot.carConfig = agent.carConfig;
		snapshot.tickSkip = agent.tickSkip;
		snapshot.actionDelay = agent.actionDelay;
		snapshot.maxTeamSize = agent.maxTeamSize;
		snapshot.ballPredTickRate = agent.ballPredTickRate;
		snapshot.useBallPrediction = agent.useBallPrediction;
		snapshot.addLayerNorm = agent.addLayerNorm;
		snapshot.optimType = agent.otimType;
		snapshot.activationType = agent.modelActivationType;
		snapshot.sharedHead = CapturePartialModelConfig(agent.sharedHead);
		snapshot.policy = CapturePartialModelConfig(agent.policy);
		snapshot.critic = CapturePartialModelConfig(agent.critic);
		return snapshot;
	}

	void ApplyAgentSetupSnapshot(Agent& agent, const AgentSetupSnapshot& snapshot)
	{
		agent.agentName = snapshot.agentName;
		agent.obsBuilderName = snapshot.obsBuilderName;
		agent.actionParserName = snapshot.actionParserName;
		agent.gameMode = snapshot.gameMode;
		agent.carConfig = snapshot.carConfig;
		agent.tickSkip = snapshot.tickSkip;
		agent.actionDelay = snapshot.actionDelay;
		agent.maxTeamSize = snapshot.maxTeamSize;
		agent.ballPredTickRate = snapshot.ballPredTickRate;
		agent.useBallPrediction = snapshot.useBallPrediction;
		agent.addLayerNorm = snapshot.addLayerNorm;
		agent.otimType = snapshot.optimType;
		agent.modelActivationType = snapshot.activationType;
		agent.sharedHead = ToPartialModelConfig(snapshot.sharedHead);
		agent.policy = ToPartialModelConfig(snapshot.policy);
		agent.critic = ToPartialModelConfig(snapshot.critic);
	}

	std::vector<AgentSetupDiff> DiffAgentSetupSnapshots(
		const AgentSetupSnapshot& saved,
		const AgentSetupSnapshot& current)
	{
		std::vector<AgentSetupDiff> diffs;
		AddDiff(diffs, "Agent Name", saved.agentName, current.agentName);
		AddDiff(diffs, "Observation Builder", saved.obsBuilderName, current.obsBuilderName);
		AddDiff(diffs, "Action Parser", saved.actionParserName, current.actionParserName);
		AddDiff(diffs, "Game Mode", FormatGameMode(saved.gameMode), FormatGameMode(current.gameMode));
		AddDiff(diffs, "Car Config", FormatCarConfig(saved.carConfig), FormatCarConfig(current.carConfig));
		AddDiff(diffs, "Tick Skip", std::to_string(saved.tickSkip), std::to_string(current.tickSkip));
		AddDiff(diffs, "Action Delay", std::to_string(saved.actionDelay), std::to_string(current.actionDelay));
		AddDiff(diffs, "Max Team Size", std::to_string(saved.maxTeamSize), std::to_string(current.maxTeamSize));
		AddDiff(diffs, "Ball Prediction Tick Rate", FormatFloat(saved.ballPredTickRate), FormatFloat(current.ballPredTickRate));
		AddDiff(diffs, "Use Ball Prediction", FormatBool(saved.useBallPrediction), FormatBool(current.useBallPrediction));
		AddDiff(diffs, "Default Layer Norm", FormatBool(saved.addLayerNorm), FormatBool(current.addLayerNorm));
		AddDiff(diffs, "Default Optimizer", FormatOptimType(saved.optimType), FormatOptimType(current.optimType));
		AddDiff(diffs, "Default Activation", FormatActivationType(saved.activationType), FormatActivationType(current.activationType));
		AddPartialModelConfigDiffs(diffs, "Shared Head", saved.sharedHead, current.sharedHead);
		AddPartialModelConfigDiffs(diffs, "Policy", saved.policy, current.policy);
		AddPartialModelConfigDiffs(diffs, "Critic", saved.critic, current.critic);
		return diffs;
	}

	bool SameAgentSetupSnapshot(const AgentSetupSnapshot& a, const AgentSetupSnapshot& b)
	{
		return DiffAgentSetupSnapshots(a, b).empty();
	}

	json AgentSetupSnapshotToJson(const AgentSetupSnapshot& snapshot)
	{
		return json{
			{ "agent_name", snapshot.agentName },
			{ "obs_builder", snapshot.obsBuilderName },
			{ "action_parser", snapshot.actionParserName },
			{ "game_mode", static_cast<int>(snapshot.gameMode) },
			{ "car_config", CarConfigToJson(snapshot.carConfig) },
			{ "tick_skip", snapshot.tickSkip },
			{ "action_delay", snapshot.actionDelay },
			{ "max_team_size", snapshot.maxTeamSize },
			{ "ball_pred_tick_rate", snapshot.ballPredTickRate },
			{ "use_ball_prediction", snapshot.useBallPrediction },
			{ "add_layer_norm", snapshot.addLayerNorm },
			{ "optim_type", static_cast<int>(snapshot.optimType) },
			{ "activation_type", static_cast<int>(snapshot.activationType) },
			{ "shared_head", PartialModelConfigToJson(snapshot.sharedHead) },
			{ "policy", PartialModelConfigToJson(snapshot.policy) },
			{ "critic", PartialModelConfigToJson(snapshot.critic) },
		};
	}

	std::optional<AgentSetupSnapshot> AgentSetupSnapshotFromJson(const json& j)
	{
		if (!j.is_object())
			return std::nullopt;
		if (!j.contains("obs_builder") || !j["obs_builder"].is_string())
			return std::nullopt;
		if (!j.contains("action_parser") || !j["action_parser"].is_string())
			return std::nullopt;

		AgentSetupSnapshot snapshot = {};
		snapshot.agentName = j.value("agent_name", std::string());
		snapshot.obsBuilderName = j["obs_builder"].get<std::string>();
		snapshot.actionParserName = j["action_parser"].get<std::string>();
		if (snapshot.obsBuilderName.empty() || snapshot.actionParserName.empty())
			return std::nullopt;
		snapshot.gameMode = static_cast<RocketSim::GameMode>(j.value("game_mode", 0));
		snapshot.carConfig = CarConfigFromJson(j.value("car_config", json::object()));
		snapshot.tickSkip = j.value("tick_skip", 8);
		snapshot.actionDelay = j.value("action_delay", snapshot.tickSkip - 1);
		snapshot.maxTeamSize = j.value("max_team_size", 1);
		snapshot.ballPredTickRate = j.value("ball_pred_tick_rate", 0.f);
		snapshot.useBallPrediction = j.value("use_ball_prediction", false);
		snapshot.addLayerNorm = j.value("add_layer_norm", true);
		snapshot.optimType = static_cast<GGL::ModelOptimType>(j.value("optim_type", 0));
		snapshot.activationType = static_cast<GGL::ModelActivationType>(j.value("activation_type", 0));
		snapshot.sharedHead = PartialModelConfigFromJson(j.value("shared_head", json::object()));
		snapshot.policy = PartialModelConfigFromJson(j.value("policy", json::object()));
		snapshot.critic = PartialModelConfigFromJson(j.value("critic", json::object()));
		return snapshot;
	}
}
