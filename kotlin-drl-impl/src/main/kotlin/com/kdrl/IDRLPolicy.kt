package com.kdrl

interface IDRLPolicy<Observation, Action> {
    fun act(observation: Observation): Action
}
