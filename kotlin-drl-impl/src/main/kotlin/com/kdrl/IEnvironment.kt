package com.kdrl

interface IEnvironment<S: IState, A: IAction> {
    /**
     * Performs an action in the environment
     */
    fun step(action: A): Step<S, A>

    /**
     * Resets the environmnent
     */
    fun reset(): S

    fun sampleAction(): A
}
