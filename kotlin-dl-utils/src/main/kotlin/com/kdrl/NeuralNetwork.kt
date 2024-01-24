package com.kdrl

import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

sealed class NeuralNetworkWrapper<T: NeuralNetwork>(val network: T) {
    abstract fun fit(data: INDArray, labels: INDArray)
    abstract fun output(input: INDArray): INDArray

    abstract fun setParams(params: INDArray)
}

class NNMultiLayerNetwork(network: MultiLayerNetwork): NeuralNetworkWrapper<MultiLayerNetwork>(network) {
    override fun fit(data: INDArray, labels: INDArray) {
        network.fit(data, labels)
    }

    override fun output(input: INDArray): INDArray {
        return network.output(input)
    }

    override fun setParams(params: INDArray) {
        network.setParams(params)
    }
}

class NNComputationGraph(network: ComputationGraph): NeuralNetworkWrapper<ComputationGraph>(network) {
    override fun fit(data: INDArray, labels: INDArray) {
        network.fit(arrayOf(data), arrayOf(labels))
    }

    override fun output(input: INDArray): INDArray {
         return Nd4j.toFlattened(*network.output(input))
    }

    override fun setParams(params: INDArray) {
        network.setParams(params)
    }

}
