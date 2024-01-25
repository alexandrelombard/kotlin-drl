package com.kdrl.space

import kotlin.random.Random

class Box<T>(
    val low: FloatArray,
    val high: FloatArray,
    val shape: IntArray, override val random: Random = Random) : ISpace<T> {
    init {
        if(low.size != high.size) {
            throw IllegalArgumentException("Low and High sizes must match")
        }

        // TODO Check that low and high dimensions are matching with the shape
    }

}
