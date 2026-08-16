#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <GigaLearnInfer/AgentComponentRegistry.h>

REGISTER_AGENT("Swishy", Agent)

#include <RLGymCPP/ActionParsers/DefaultAction.h>
REGISTER_ACTION_PARSER("DefaultAction", [](const Agent&) {
    return std::make_unique<RLGC::DefaultAction>();
})

