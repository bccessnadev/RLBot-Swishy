#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentComponentRegistry.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

using namespace RocketSim;
using namespace RLGC; // RLGymCPP

Agent::Agent()
{
	agentName = "agent";
	obsBuilderName = "DefaultObsPadded";
	actionParserName = "DefaultAction";
	gameMode = GameMode::SOCCAR;
	sharedHead = {};
	sharedHead.layerSizes = { 256 };
	sharedHead.addOutputLayer = false;
	policy = {};
	policy.layerSizes = { 256, 256, 256 };
	critic = {};
	critic.layerSizes = { 256, 256, 256 };
}

ObsBuilder* Agent::MakeObsBuilder() const
{
	const std::string name = obsBuilderName.empty() ? "DefaultObsPadded" : obsBuilderName;
	return ObsBuilderRegistry::Create(name, *this).release();
}

int Agent::GetObsSize() const 
{ 
	int obsSize = 0;

	if (GetStage() == RocketSimStage::UNINITIALIZED)
		Init(std::filesystem::path(PROJECT_ROOT) / "ForgeTraining" / "collision_meshes", true);

	// Construct a temporary environment to build a Obs from, then return the created size
	// Makes it more convinient to not need to figure this out manually for each Obs
	if (Arena* arena = Arena::Create(gameMode))
	{
		for (int i = 0; i < maxTeamSize; i++) {
			arena->AddCar(Team::BLUE, carConfig);
			arena->AddCar(Team::ORANGE, carConfig);
		}

		GameState tempGameState = GameState(arena);

		if (ObsBuilder* tempObsBuilder = MakeObsBuilder())
		{
			if (tempGameState.players.size() > 0)
			{
				tempObsBuilder->Reset(tempGameState);
				FList tempObs = tempObsBuilder->BuildObs(tempGameState.players[0], tempGameState);
				obsSize = tempObs.size();
			}

			delete tempObsBuilder;
		}

		delete arena;
	}

	return obsSize;
}

ActionParser* Agent::MakeActionParser() const
{
	const std::string name = actionParserName.empty() ? "DefaultAction" : actionParserName;
	return ActionParserRegistry::Create(name, *this).release();
}

std::unique_ptr<Agent> Agent::Clone() const { return std::make_unique<Agent>(*this); }

void Agent::MakeInferenceModels(
	int obsSize,
	int numActions,
	const c10::Device& device,
	GGL::ModelSet& outModels) const
{
	GGL::ModelManager::MakeModels(
		false,
		obsSize,
		numActions,
		sharedHead,
		policy,
		{},
		device,
		outModels);
}