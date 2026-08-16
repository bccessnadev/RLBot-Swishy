#include "SwishyObs.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentComponentRegistry.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>

using namespace RocketSim;
using namespace std;

namespace RLGC
{
	// Expresses a world-space point in the local frame defined by rot and origin.
	static Vec ToLocalSpace(const RotMat& rot, const Vec& point, const Vec& origin) {
		return rot.Dot(point - origin);
	}
	
	REGISTER_OBS_BUILDER("SwishyObs", [](const Agent& agent) {
	return std::make_unique<RLGC::SwishyObs>(agent.maxTeamSize);
	})

	void SwishyObs::Reset(const GameState& initialState)
	{
		// Request ball state for the maximum prediction time to initialize ball prediction.
		if (!ballPredTimes.empty())
			initialState.GetFutureBallState(ballPredTimes[ballPredTimes.size() - 1]);
	}

	void SwishyObs::AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) const
	{
		const PhysState phys = InvertPhys(player, inv);

		obs += phys.pos * POS_COEF;
		obs += phys.rotMat.forward;
		obs += phys.rotMat.up;
		obs += phys.vel * VEL_COEF;
		obs += phys.angVel * ANG_VEL_COEF;                  // world-frame angular velocity
		obs += phys.rotMat.Dot(phys.angVel) * ANG_VEL_COEF; // local-frame angular velocity

		obs += ToLocalSpace(phys.rotMat, ball.pos, phys.pos) * POS_COEF; // local ball pos
		obs += ToLocalSpace(phys.rotMat, ball.vel, phys.vel) * VEL_COEF; // local ball vel

		obs += player.boost / 100;
		obs += player.isOnGround;
		obs += player.HasFlipOrJump();
		obs += player.isDemoed;
		obs += player.hasJumped;
	}

	FList SwishyObs::BuildObs(const Player& player, const GameState& state)
	{
		const bool inv = player.team == Team::ORANGE;

		FList out;

		const PhysState ball = InvertPhys(state.ball, inv);
		auto& pads = state.GetBoostPads(inv);
		auto& padTimers = state.GetBoostPadTimers(inv);

		// Current ball state
		out += ball.pos * POS_COEF;
		out += ball.vel * VEL_COEF;
		out += ball.angVel * ANG_VEL_COEF;

		// Previous actions
		for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
			out += player.prevAction[i];

		// Boost pad states, blended by timer (approaches 1 as the pad becomes available)
		for (int i = 0; i < state.GetBoostLocationAmount(); i++) {
			if (pads[i])
				out += 1.f;
			else
				out += 1.f / (1.f + padTimers[i]);
		}

		// Self slot (built into its own list so we know the fixed per-player slot size for padding)
		FList selfObs;
		AddPlayerToObs(selfObs, player, inv, ball);
		const int playerObsSize = (int)selfObs.size();
		out += selfObs;

		// Other players, split into teammates and opponents
		std::vector<FList> teammates, opponents;
		for (const Player& otherPlayer : state.players) {
			if (otherPlayer.carId == player.carId)
				continue;

			FList playerObs;
			AddPlayerToObs(playerObs, otherPlayer, inv, ball);
			((otherPlayer.team == player.team) ? teammates : opponents).push_back(std::move(playerObs));
		}

		if ((int)teammates.size() > maxPlayers - 1)
			RG_ERR_CLOSE("SwishyObs: Too many teammates for Obs, maximum is " << (maxPlayers - 1));
		if ((int)opponents.size() > maxPlayers)
			RG_ERR_CLOSE("SwishyObs: Too many opponents for Obs, maximum is " << maxPlayers);

		// Zero-pad to fixed counts, then shuffle (matches training-time ordering distribution)
		while ((int)teammates.size() < maxPlayers - 1)
			teammates.push_back(FList(playerObsSize));
		while ((int)opponents.size() < maxPlayers)
			opponents.push_back(FList(playerObsSize));

		std::shuffle(teammates.begin(), teammates.end(), ::Math::GetRandEngine());
		std::shuffle(opponents.begin(), opponents.end(), ::Math::GetRandEngine());

		for (const FList& teammate : teammates)
			out += teammate;
		for (const FList& opponent : opponents)
			out += opponent;

		// Ball prediction: inverted world-frame pos/vel/angVel per predicted time
		for (float predTime : ballPredTimes) {
			const PhysState predBall = InvertPhys(state.GetFutureBallState(predTime), inv);
			out += predBall.pos * POS_COEF;
			out += predBall.vel * VEL_COEF;
			out += predBall.angVel * ANG_VEL_COEF;
		}

		return out;
	}
}
