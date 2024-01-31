package com.kdrl

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nadam
import kotlin.test.Test


class MultiLayerNetworkExternalErrorsTest {
    @Test
    fun testExternalErrors() {
        //Create the model

        val nIn = 4
        val nOut = 3
        Nd4j.getRandom().setSeed(12345)
        val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(Nadam())
            .list()
            .layer(DenseLayer.Builder().nIn(nIn).nOut(3).build())
            .layer(DenseLayer.Builder().nIn(3).nOut(3).build())
            .build()

        val model = MultiLayerNetwork(conf)
        model.init()

        for(i in 0..100) {
            //Calculate gradient with respect to an external error
            val minibatch = 32
            val input = Nd4j.rand(minibatch, nIn)
            model.input = input
            //Do forward pass, but don't clear the input activations in each layers - we need those set so we can calculate
            // gradients based on them
            model.feedForward(true, false)

            val externalError = Nd4j.rand(minibatch, nOut)
            val p = model.backpropGradient(externalError, null) //Calculate backprop gradient based on error array

            //Update the gradient: apply learning rate, momentum, etc
            //This modifies the Gradient object in-place
            val gradient: Gradient = p.getFirst()
            val iteration = 0
            val epoch = 0
            model.updater.update(model, gradient, iteration, epoch, minibatch, LayerWorkspaceMgr.noWorkspaces())

            //Get a row vector gradient array, and apply it to the parameters to update the model
            val updateVector = gradient.gradient()
            model.params().subi(updateVector)
        }

    }

}
