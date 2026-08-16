#include "RLBotPolicyRuntime.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/InferUnit.h>
#include <RLGymCPP/Framework.h>

#include <chrono>

using namespace RLGC;

RLBotPolicyRuntime::RLBotPolicyRuntime(
    const Agent& agent,
    const std::filesystem::path& modelsFolder,
    bool useGPU,
    RLBotPolicyObserver* observer)
    : observer_(observer)
{
    // Complete model loading and one live-shaped forward pass before RLBot sends
    // InitComplete. Large CUDA policies otherwise stall their first packet.
    const auto startup = std::chrono::steady_clock::now();
    obsBuilder_.reset(agent.MakeObsBuilder());
    actionParser_.reset(agent.MakeActionParser());
    const int obsSize = agent.GetObsSize();
    inferUnit_ = std::make_unique<GGL::InferUnit>(
        obsBuilder_.get(),
        obsSize,
        actionParser_.get(),
        agent.sharedHead,
        agent.policy,
        modelsFolder,
        useGPU);

    if (RocketSim::Arena* warmupArena = RocketSim::Arena::Create(agent.gameMode)) {
        startupWarmupArena_.reset(warmupArena);
        warmupArena->AddCar(RocketSim::Team::BLUE, agent.carConfig);
        warmupArena->AddCar(RocketSim::Team::ORANGE, agent.carConfig);

        GameState warmupState(warmupArena);
        obsBuilder_->Reset(warmupState);
		inferUnit_->InferActionIndex(warmupState.players.front(), warmupState, true);
    }

    inferUnit_->captureDiagnostics = observer_ && observer_->CapturePolicyDiagnostics();
    const double startupMicros = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - startup).count();
    if (observer_)
        observer_->OnPolicyRuntimeReady(startupMicros);

    RG_LOG("RLBot runtime ready: agent=" << agent.agentName
        << " obs_builder=" << agent.obsBuilderName
        << " action_parser=" << agent.actionParserName
        << " obs=" << obsSize
        << " actions=" << actionParser_->GetActionAmount()
        << " tick_skip=" << agent.tickSkip
        << " action_delay=" << agent.actionDelay
        << " device=" << (useGPU ? "GPU" : "CPU")
        << " model=" << modelsFolder
        << " startup_ms=" << startupMicros / 1000.0);
}

RLBotPolicyRuntime::~RLBotPolicyRuntime() = default;

RLGC::ActionParser& RLBotPolicyRuntime::GetActionParser()
{
    return *actionParser_;
}

void RLBotPolicyRuntime::RequestObservationReset()
{
    observationNeedsReset_ = true;
}

void RLBotPolicyRuntime::PrepareObservationState(RLGC::GameState& gameState)
{
    if (!observationNeedsReset_)
        return;

    obsBuilder_->Reset(gameState);
    startupWarmupArena_.reset();
    observationNeedsReset_ = false;
}

int RLBotPolicyRuntime::Infer(
    const RLGC::Player& player,
    const RLGC::GameState& gameState)
{
    inferUnit_->captureDiagnostics = observer_ && observer_->CapturePolicyDiagnostics();
    const auto inferenceStart = observer_
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const int actionId = inferUnit_->InferActionIndex(player, gameState, true);
    if (observer_) {
        const double inferenceMicros = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - inferenceStart).count();
        observer_->OnPolicyInference(actionId, inferenceMicros, *inferUnit_);
    }
    return actionId;
}
