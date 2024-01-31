package com.kdrl

import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

sealed class NeuralNetworkWrapper<T: NeuralNetwork>(val network: T) {
    fun init() {
        network.init()
    }

    fun params(): INDArray {
        return network.params()
    }

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

fun MultiLayerNetwork.wrap(): NNMultiLayerNetwork {
    return NNMultiLayerNetwork(this)
}

fun MultiLayerNetwork.updateWithExternalError(input: INDArray, error: INDArray, minibatchSize: Int = 32) {
    this.input = input
    this.feedForward(true, false)
    val p = this.backpropGradient(error, null)
    val gradient = p.first
    this.updater.update(this, gradient, 0, 0, minibatchSize, LayerWorkspaceMgr.noWorkspaces())
    val updateVector = gradient.gradient()
    this.params().subi(updateVector)
}

class NNComputationGraph(network: ComputationGraph): NeuralNetworkWrapper<ComputationGraph>(network) {
    override fun fit(data: INDArray, labels: INDArray) {
        network.fit(arrayOf(data), arrayOf(labels))
    }

    override fun output(input: INDArray): INDArray {
         return network.outputSingle(input)
    }

    override fun setParams(params: INDArray) {
        network.setParams(params)
    }
}

fun ComputationGraph.wrap(): NNComputationGraph {
    return NNComputationGraph(this)
}
