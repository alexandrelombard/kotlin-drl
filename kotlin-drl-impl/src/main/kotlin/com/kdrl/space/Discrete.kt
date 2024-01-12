package com.kdrl.space

import kotlin.random.Random

class Discrete(val size: Int, override val random: Random = Random): ISpace {
    fun sample(): Int {
        return this.random.nextInt(size)
    }
}
