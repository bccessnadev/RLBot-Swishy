#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <vector>

namespace RLGC
{
	/**
	 * Swishy's observation builder.
	 * An adaption of the AdvancedObsPadded with added ball prediction
	 */
	class SwishyObs : public ObsBuilder {
	public:

		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f;

		int maxPlayers = 2;

		SwishyObs(const int maxPlayers = 2, const std::vector<float>& ballPredTimes = { 0.5f, 1.f, 3.f })
			: maxPlayers(maxPlayers), ballPredTimes(ballPredTimes) {}

		void Reset(const GameState& initialState) override;

		FList BuildObs(const Player& player, const GameState& state) override;
		
		void AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) const;

	protected:

		/** A ball is predicted into the future for each time in this list. */
		std::vector<float> ballPredTimes;
	};
}
