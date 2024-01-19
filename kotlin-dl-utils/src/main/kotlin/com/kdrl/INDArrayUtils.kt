package com.kdrl

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun FloatArray.toINDArray(): INDArray {
    return Nd4j.create(this)
}

fun IntArray.toINDArray(): INDArray {
    return Nd4j.createFromArray(*this)
}

fun BooleanArray.toINDArray(): INDArray {
    return Nd4j.create(this)
}

fun List<FloatArray>.toINDArray(): INDArray {
    return Nd4j.create(this.toTypedArray())
}

operator fun INDArray.plus(i: INDArray): INDArray {
    return this.add(i)
}

operator fun INDArray.minus(i: INDArray): INDArray {
    return this.sub(i)
}

operator fun INDArray.times(n: Number): INDArray {
    return this.mul(n)
}

operator fun Number.times(i: INDArray): INDArray {
    return i.mul(this)
}
