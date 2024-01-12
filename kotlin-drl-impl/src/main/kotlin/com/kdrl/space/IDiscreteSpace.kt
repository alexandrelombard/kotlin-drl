package com.kdrl.space

interface IDiscreteSpace : ISpace<Int> {
    val size: Int

    fun sample(): Int {
        return random.nextInt(size)
    }
}
