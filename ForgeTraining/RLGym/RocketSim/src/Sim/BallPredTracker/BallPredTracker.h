#pragma once
#include "../Arena/Arena.h"

RS_NS_START

/**
 * Fixed-capacity ring buffer for predicted ball states.
 * External readers see a sequential view: indices [0, size()) return states
 * in chronological order starting from the most recent "now" tick.
 **/
struct BallPredBuffer {
	/** Backing storage; sized once by Reset() and never resized afterwards. **/
	std::vector<BallState> data;

	/** Index of the logical element [0] in data. **/
	size_t head = 0;

	/** Number of valid elements currently in the buffer. **/
	size_t count = 0;

	void Reset(size_t capacity) {
		data.assign(capacity, BallState());
		head = 0;
		count = 0;
	}

	size_t size() const     { return count; }
	size_t capacity() const { return data.size(); }
	bool   empty() const    { return count == 0; }

	const BallState& operator[](size_t i) const { return data[(head + i) % data.size()]; }
	BallState&       operator[](size_t i)       { return data[(head + i) % data.size()]; }
	const BallState& front() const { return (*this)[0]; }
	const BallState& back()  const { return (*this)[count - 1]; }

	/** Precondition: count < capacity. **/
	void Push(const BallState& s) {
		data[(head + count) % data.size()] = s;
		++count;
	}

	/** Precondition: n <= count. **/
	void Advance(size_t n) {
		head = (head + n) % data.size();
		count -= n;
	}
};

// An external tool struct that predicts the ball of a given arena
struct BallPredTracker {
	Arena* ballPredArena;
	BallPredBuffer predData;
	size_t numPredTicks;

	int lastUpdateTickCount;

	// Keep track earliest prediction of ball being scored
	int predScoreStepIndex;

	// Cached tick-rate ratio so we don't divide every call. Set in ctor.
	float mainToPredTickRatio = 1.f;

	// Fractional pred-tick remainder from the previous UpdatePredManual call.
	// Accumulates sub-tick remainders so they are eventually credited.
	float _pendingPredTickFraction = 0.f;

	// BallState::Matches() tolerances pre-computed in the constructor.
	// At ratio=1 (pred rate == main rate) these equal the BallState defaults.
	// At lower pred rates they are expanded to absorb the collision-detection
	// timing drift that arises when the two arenas use different timesteps.
	// timingError = (1 - mainToPredTickRatio) / predRate  (seconds/pred-step)
	//
	// pos:    4 * maxBallSpeed * timingError  — handles two-bounce windows
	// vel:    1.5 * (2 * v² / R) * timingError  — curved-surface normal divergence
	//         (R = 520 uu, tightest RL corner; catches car contacts > ~60 uu/s)
	// angVel: 2.0 rad/s empirical  — spin-friction divergence at bounces
	float _matchMarginPos    = 0.8f;
	float _matchMarginVel    = 0.4f;
	float _matchMarginAngVel = 0.02f;

	// arena: The arena you want to predict the ball for (BallPredTracker will make a copy of it without the cars)
	// You do not need to make another arena for BallPredTracker, it does that itself
	BallPredTracker(Arena* arena, size_t numPredTicks);
	~BallPredTracker();

	// No copying
	BallPredTracker(const BallPredTracker& other) = delete;
	BallPredTracker& operator=(const BallPredTracker& other) = delete;

	// Allow for changing how many ticks to predict. Helpful for shared prediction tracker
	void SetNumPredTicks(size_t newNumPredTicks);

	// Update the prediction data from the arena the ball is in, does not need to be called every tick
	// The arena is needed for the current ball state, as well as the tick count to determine time since last update
	void UpdatePredFromArena(Arena* arena);

	// An alternate version of UpdatePred which doesn't require the arena, 
	//	but instead you manually provide the current ball state and the ticks since this tracker was last updated
	void UpdatePredManual(const BallState& curBallState, int ticksSinceLastUpdate);

	// Forcefully re-predicts all ticks
	void ForceUpdateAllPred(const BallState& initialBallState);

	// Called after each ballPredArena to update predictions
	void PostStepUpdate();

	// Get the predicted ball state at a given future time delta
	BallState GetBallStateForTime(float predTime) const;
};

RS_NS_END