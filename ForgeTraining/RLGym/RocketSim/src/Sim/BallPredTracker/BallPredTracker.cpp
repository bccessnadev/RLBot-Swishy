#include "BallPredTracker.h"

RS_NS_START

BallPredTracker::BallPredTracker(Arena* arena, size_t numPredTicks) : numPredTicks(numPredTicks) {
	const float mainRate = arena->GetTickRate();
	const float configRate = arena->GetArenaConfig().ballPredTickRate;
	const float predRate = (configRate > 0.f) ? configRate : mainRate;

	ballPredArena = Arena::Create(arena->gameMode, arena->GetArenaConfig(), predRate);
	mainToPredTickRatio = predRate / mainRate;

	// Pre-compute Matches() tolerances scaled for the timing error introduced
	// when the pred arena runs at a different rate than the main arena.
	// timingError = (1 - ratio) / predRate is the maximum duration (in seconds)
	// by which collision detection can lag between the two arenas per pred step.
	if (mainToPredTickRatio < 1.f) {
		const float timingError = (1.f - mainToPredTickRatio) / predRate;

		// --- Position ---
		// A bounce detected timingError seconds late causes a position error of
		// |v_pre - v_post| * timingError ≤ 2 * maxBallSpeed * timingError per bounce.
		// Use 4× to handle two-bounce windows (e.g. corner rattles).
		_matchMarginPos = 0.8f + 4.f * 2300.f * timingError;

		// --- Velocity ---
		// On flat surfaces the only post-bounce velocity error is gravity * timingError (~5 uu/s).
		// On curved surfaces (RL corners, R ≈ 256 uu) the different bounce position means a
		// different surface normal, causing velocity divergence of up to:
		//   Δv ≈ 2 * v² * timingError / R ≈ 170 uu/s at 60Hz/120Hz
		// Add 1.5× safety factor for compound effects (high spin, multi-bounce).
		// At 60Hz/120Hz this gives ~255 uu/s, which still catches any car contact
		// where the car is moving faster than ~60 uu/s relative to the ball.
		static constexpr float RS_MIN_CORNER_RADIUS = 256.f;
		_matchMarginVel = 0.4f + 1.5f * (2.f * 2300.f * 2300.f / RS_MIN_CORNER_RADIUS) * timingError;

		// --- Angular velocity ---
		// Spin-friction divergence at bounces, empirically observed < 1 rad/s.
		// Use 2 rad/s to cover high-spin scenarios while still catching car contacts (> 2 rad/s).
		_matchMarginAngVel = 2.0f;
	}

	lastUpdateTickCount = 0;
	predScoreStepIndex = -1;

	predData.Reset(numPredTicks);
	UpdatePredFromArena(arena);
}

BallPredTracker::~BallPredTracker() {
	delete ballPredArena;
}

void BallPredTracker::SetNumPredTicks(size_t newNumPredTicks)
{
	if (newNumPredTicks == numPredTicks)
		return;

	numPredTicks = newNumPredTicks;
	predScoreStepIndex = -1;

	if (!predData.empty()) {
		BallState seed = predData[0];
		ForceUpdateAllPred(seed);
	} else {
		// Buffer is uninitialized; Reset allocates storage so it will be
		// populated on the next UpdatePredManual / UpdatePredFromArena call.
		predData.Reset(numPredTicks);
	}
}

void BallPredTracker::UpdatePredFromArena(Arena* arena) {
	BallState bs = arena->ball->GetState();

	int ticksSinceLastUpdate = arena->tickCount - lastUpdateTickCount;
	UpdatePredManual(bs, ticksSinceLastUpdate);
}

void BallPredTracker::PostStepUpdate()
{
	predData.Push(ballPredArena->ball->GetState());
	const int stepIndex = static_cast<int>(predData.size()) - 1;

	// Track earliest tick where the ball scores.
	//
	// IsBallScored() is purely positional, and the prediction arena never resets on a
	// goal, so once the predicted ball crosses the goal line it keeps flying into the
	// net and eventually bounces back out — making IsBallScored() false again for that
	// same shot. A ball that has fully crossed the line has scored and cannot un-score,
	// so we only ever record the earliest crossing here and never clear it on a later
	// non-scoring step. Stale indices are handled elsewhere: ForceUpdateAllPred() resets
	// to -1 before re-sweeping, and the incremental advance in UpdatePredManual() shifts
	// the index forward and invalidates it once it scrolls off the front of the buffer.
	if (ballPredArena->IsBallScored()) {
		if (stepIndex < predScoreStepIndex || predScoreStepIndex < 0)
			predScoreStepIndex = stepIndex;
	}
}

void BallPredTracker::UpdatePredManual(const BallState& curBallState, int ticksSinceLastUpdate) {

	// Credit any fractional remainder from previous calls so sub-tick amounts
	// are eventually flushed rather than silently dropped each update.
	const float predTicksExact = ticksSinceLastUpdate * mainToPredTickRatio + _pendingPredTickFraction;
	const int predTicksSinceLastUpdate = static_cast<int>(predTicksExact);
	_pendingPredTickFraction = predTicksExact - static_cast<float>(predTicksSinceLastUpdate);

	bool needsFullRepred;
	if (predTicksSinceLastUpdate == 0) {
		// Not enough main-arena ticks have elapsed to complete one pred tick.
		// Keep the existing prediction; force a repred only if the buffer is uninitialized.
		needsFullRepred = predData.empty();
	} else if (static_cast<size_t>(predTicksSinceLastUpdate) < predData.size()) {

		if (predData[predTicksSinceLastUpdate].Matches(curBallState, _matchMarginPos, _matchMarginVel, _matchMarginAngVel)) {
			// We can re-use ball prediction data
			needsFullRepred = false;

			// O(1) head advance instead of O(N) vector::erase + memmove.
			predData.Advance(static_cast<size_t>(predTicksSinceLastUpdate));

			// Shift the score index by the same amount; invalidate if it
			// falls before the new logical [0].
			if (predScoreStepIndex >= 0) {
				predScoreStepIndex -= predTicksSinceLastUpdate;
				if (predScoreStepIndex < 0)
					predScoreStepIndex = -1;
			}

			// Predict new states until we reach numPredTicks
			ballPredArena->ball->SetState(predData.back());
			while (predData.size() < numPredTicks) {
				ballPredArena->Step(1);
				PostStepUpdate();
			}
		} else {
			needsFullRepred = true;
		}
	} else {
		needsFullRepred = true;
	}

	if (needsFullRepred) {
		ForceUpdateAllPred(curBallState);
	}

	// lastUpdateTickCount is kept in MAIN-arena ticks (matches caller).
	lastUpdateTickCount += ticksSinceLastUpdate;
}

void BallPredTracker::ForceUpdateAllPred(const BallState& initialBallState) {
	ballPredArena->ball->SetState(initialBallState);

	predData.Reset(numPredTicks);
	predScoreStepIndex = -1;
	_pendingPredTickFraction = 0.f;

	predData.Push(initialBallState);

	if (ballPredArena->IsBallScored() && predScoreStepIndex < 0)
		predScoreStepIndex = 0;

	for (size_t i = 1; i < numPredTicks; i++) {
		ballPredArena->Step();
		PostStepUpdate();
	}
}

BallState BallPredTracker::GetBallStateForTime(float predTime) const {
	if (predData.empty())
		RS_ERR_CLOSE("BallPredTracker::GetBallStateForTime(): Predicted ball data is empty, update prediction before calling");

	int index = RS_CLAMP(predTime / ballPredArena->tickTime, 0, predData.size() - 1);
	return predData[index];
}

RS_NS_END
