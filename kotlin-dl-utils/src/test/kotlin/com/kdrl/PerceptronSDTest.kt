package com.kdrl

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.weightinit.impl.XavierInitScheme
import kotlin.random.Random
import kotlin.test.Test

class PerceptronSDTest {
    @Test
    fun testPerceptronWithSameDiff() {
        val sd = SameDiff.create()

        val nIn = 2L
        val nOut = 1L
        val input = sd.placeHolder("input", DataType.FLOAT, -1, nIn)
        val output = sd.placeHolder("output", DataType.FLOAT, -1, nOut)

        val w0 = sd.`var`("w0", XavierInitScheme('c', nIn.toDouble(), nOut.toDouble()), DataType.FLOAT, nIn, 128)
        val b0 = sd.`var`("bias", 1, 128)
        val a0 = sd.nn().linear(input, w0, b0)

        val w1 = sd.`var`("w1", XavierInitScheme('c', nIn.toDouble(), nOut.toDouble()), DataType.FLOAT, nIn, 128)
        val b1 = sd.`var`("bias", 1, nOut)
        val a1 = sd.nn().linear(a0, w1, b1)

        val diff = sd.math.squaredDifference("prediction", a1, output)
        val loss = diff.mean()

        sd.setLossVariables(loss)

        // Building the XOR dataset
        val trainingSetSize = 100000
        val features = Nd4j.create(trainingSetSize, 2)
        val labels = Nd4j.create(trainingSetSize, 1)

        for(i in 0 until trainingSetSize) {
            val f1 = Random.nextFloat() * 100
            val f2 = Random.nextFloat() * 100
            val l = f1 * 5 + f2

            features.putScalar(intArrayOf(i, 0), f1)
            features.putScalar(intArrayOf(i, 1), f2)
            labels.putScalar(intArrayOf(i, 0), l)
        }

        // Train
        sd.fit()
    }
}
