#pragma once

namespace GGL { struct InferUnit; }

/** Receives optional policy telemetry without coupling the runtime to development tooling. **/
class RLBotPolicyObserver {
public:
    virtual ~RLBotPolicyObserver() = default;

    virtual bool CapturePolicyDiagnostics() const = 0;
    virtual void OnPolicyRuntimeReady(double startupMicros) = 0;
    virtual void OnPolicyInference(
        int actionId,
        double inferenceMicros,
        const GGL::InferUnit& inferUnit) = 0;
};
