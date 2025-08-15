export default class NN {
    /**
     * Constructs a new Neural Network with a fixed architecture.
     * @param {number[]} layerLengths An array where each element is the number of neurons in that layer.
     * e.g., [input_size, hidden1_size, hidden2_size, ..., output_size]
     */
    constructor(layerLengths) {
        this.layerLengths = layerLengths;

        this.totalNeurons = 0;
        this.layerOffsets = new Uint32Array(layerLengths.length);
        let currentOffset = 0;
        for (let i = 0; i < layerLengths.length; i++) {
            this.layerOffsets[i] = currentOffset;
            this.totalNeurons += layerLengths[i];
            currentOffset += layerLengths[i];
        }
        this.layerValues = new Float32Array(this.totalNeurons);
        this.layerSums = new Float32Array(this.totalNeurons); // For storing pre-activation sums (z-values)

        let totalWeightsCount = 0;
        for (let i = 0; i < layerLengths.length - 1; i++) {
            totalWeightsCount += layerLengths[i] * layerLengths[i + 1];
        }
        this.weights = new Float32Array(totalWeightsCount);

        let totalBiasesCount = 0;
        for (let i = 1; i < layerLengths.length; i++) {
            totalBiasesCount += layerLengths[i];
        }
        this.biases = new Float32Array(totalBiasesCount);

        // Pre-calculate offsets for faster indexing
        this.weightOffsets = new Uint32Array(layerLengths.length - 1);
        let currentWeightOffset = 0;
        for (let i = 0; i < layerLengths.length - 1; i++) {
            this.weightOffsets[i] = currentWeightOffset;
            currentWeightOffset += layerLengths[i] * layerLengths[i + 1];
        }

        this.biasOffsets = new Uint32Array(layerLengths.length);
        this.biasOffsets[0] = 0; // Layer 0 has no biases.
        let currentBiasOffset = 0;
        for (let i = 1; i < layerLengths.length; i++) {
            this.biasOffsets[i] = currentBiasOffset;
            currentBiasOffset += layerLengths[i];
        }

        this.weightVelocities = new Float32Array(this.weights.length);
        this.biasVelocities = new Float32Array(this.biases.length);
        this.weightGrad = new Float32Array(this.weights.length);
        this.biasGrad = new Float32Array(this.biases.length);
        this.errors = new Float32Array(this.totalNeurons);
        this._randomizeParameters(0.1);
    }

    /**
     * Initializes weights and biases with random values within a given range.
     * @param {number} range The maximum absolute value for random initialization (e.g., 0.1 for values between -0.1 and 0.1)
     */
    _randomizeParameters(range) {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = this.guassian(range);
        }
        for (let i = 0; i < this.biases.length; i++) {
            this.biases[i] = this.guassian(range);
        }
    }

    /**
     * Sigmoid activation function.
     * @param {number} x The input value.
     * @returns {number} The activated value.
     */
    _sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    _linear(x) {
        return x;
    }

    /**
     * Tanh (Hyperbolic Tangent) activation function.
     * @param {number} x The input value.
     * @returns {number} The activated value.
     */
    _tanh(x) {
        return Math.tanh(x);
    }

    /**
     * Rectified Linear Unit (ReLU) activation function.
     * @param {number} x The input value.
     * @returns {number} The activated value.
     */
    _relu(x) {
        return Math.max(0, x);
    }

    /**
     * Derivative of the ReLU activation function.
     * @param {number} x The pre-activation value (z).
     * @returns {number} The derivative value (1 or 0).
     */
    _reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    _linearDerivative(x) {
        return 1;
    }

    _sigmoidDerivative(x) {
        return this._sigmoid(x) * (1 - this._sigmoid(x));
    }

    /**
     * Performs a feedforward pass through the network.
     * @param {number[]} inputs An array of input values for the network.
     * @returns {Float32Array} An array of output values from the network.
     */
    feedForward(inputs) {
        const inputLayerSize = this.layerLengths[0];
        if (inputs.length !== inputLayerSize) {
            throw new Error(`Input array length mismatch. Expected ${inputLayerSize}, got ${inputs.length}.`);
        }

        let currentLayerOffset = this.layerOffsets[0];
        for (let i = 0; i < inputLayerSize; i++) {
            this.layerValues[currentLayerOffset + i] = inputs[i];
        }

        let weightIdx = 0;
        let biasIdx = 0;

        for (let l = 0; l < this.layerLengths.length - 1; l++) {
            const currentLayerLength = this.layerLengths[l];
            const nextLayerLength = this.layerLengths[l + 1];
            const nextLayerOffset = this.layerOffsets[l + 1];
            currentLayerOffset = this.layerOffsets[l];

            for (let j = 0; j < nextLayerLength; j++) {
                let sum = 0;
                for (let i = 0; i < currentLayerLength; i++) {
                    sum += this.layerValues[currentLayerOffset + i] * this.weights[weightIdx++];
                }
                sum += this.biases[biasIdx++];

                this.layerSums[nextLayerOffset + j] = sum; // Store pre-activation sum

                if (l < this.layerLengths.length - 2) {
                    this.layerValues[nextLayerOffset + j] = this._relu(sum);
                } else {
                    this.layerValues[nextLayerOffset + j] = this._sigmoid(sum);
                }
            }
        }

        const outputLayerOffset = this.layerOffsets[this.layerLengths.length - 1];
        const outputLength = this.layerLengths[this.layerLengths.length - 1];
        const outputs = new Float32Array(outputLength);
        for (let i = 0; i < outputLength; i++) {
            outputs[i] = this.layerValues[outputLayerOffset + i];
        }
        return outputs;
    }

    /**
     * Loads weights and biases from a flat array, typically a GA chromosome.
     * Resets the Adam optimizer state.
     * @param {number[] | Float32Array} chromosomeValues A flat array containing all weights followed by all biases.
     */
    loadChromosome(chromosomeValues) {
        if (chromosomeValues.length !== this.weights.length + this.biases.length) {
            throw new Error(`Chromosome length mismatch. Expected ${this.weights.length + this.biases.length}, got ${chromosomeValues.length}.`);
        }
        let currentIdx = 0;
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = chromosomeValues[currentIdx++];
        }
        for (let i = 0; i < this.biases.length; i++) {
            this.biases[i] = chromosomeValues[currentIdx++];
        }
        // Reset optimizer state upon loading new parameters
        this.weightVelocities.fill(0);
        this.biasVelocities.fill(0);
    }

    /**
     * Extracts all weights and biases into a single flat array, typically for a GA chromosome.
     * @returns {Float32Array} A flat array containing all weights followed by all biases.
     */
    getChromosome() {
        const chromosome = new Float32Array(this.weights.length + this.biases.length);
        let currentIdx = 0;
        for (let i = 0; i < this.weights.length; i++) {
            chromosome[currentIdx++] = this.weights[i];
        }
        for (let i = 0; i < this.biases.length; i++) {
            chromosome[currentIdx++] = this.biases[i];
        }
        return chromosome;
    }

    mutateWeights(mutationRate, muattionAmount) {
        for (let i = 0; i < this.weights.length; i++) {
            if (Math.random() < mutationRate) {
                this.weights[i] += this.guassian(muattionAmount);
            }
        }
    }

    mutateBiases(mutationRate, mutationAmount) {
        for (let i = 0; i < this.biases.length; i++) {
            if (Math.random() < mutationRate) {
                this.biases[i] += this.guassian(mutationAmount);
            }
        }
    }

    guassian(strength = 1) {
        const u = 1 - Math.random();
        const v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * strength;
    }

    random(strength = 1) {
        return (Math.random() * 2 - 1) * strength;
    }

    static crossOverChromosomes(c1, c2) {
        const index = Math.floor(Math.random() * c1.length);
        const newChromosome = new Float32Array(c1.length);
        for (let i = 0; i < index; i++) {
            newChromosome[i] = c1[i];
        }
        for (let i = index; i < c1.length; i++) {
            newChromosome[i] = c2[i];
        }
        return newChromosome;
    }

    export() {
        return JSON.stringify(Array.from(this.getChromosome()));
    }

    import(txt) {
        this.loadChromosome(JSON.parse(txt));
    }

    getLoss(correctOutput, output) {
        let loss = 0;
        for (let i = 0; i < correctOutput.length; i++) {
            loss += (correctOutput[i] - output[i]) * (correctOutput[i] - output[i]);
        }
        return loss;
    }

    mse(predictions, targets) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const error = predictions[i] - targets[i];
            sum += error * error;
        }
        return sum / predictions.length;
    }

    train(inputs, expectedOutputs, epochs, learningRate = 0.001) {
        const weightGrad = this.weightGrad;
        const biasGrad = this.biasGrad;
        const errors = this.errors;
        // weightGrad.fill(0);
        // biasGrad.fill(0);
        // errors.fill(0);
        let loss = 0;
        for (let epoch = 0; epoch < epochs; epoch++) {
            loss = 0;
            weightGrad.fill(0);
            biasGrad.fill(0);

            for (let inp = 0; inp < inputs.length; inp++) {
                const input = inputs[inp];
                const expectedOutput = expectedOutputs[inp];
                const actualOutput = this.feedForward(input);
                loss += this.getLoss(expectedOutput, actualOutput);

                // last layer
                const lastLayerOffset = this.layerOffsets[this.layerLengths.length - 1];
                const lastLayerLength = this.layerLengths[this.layerLengths.length - 1];
                for (let j = 0; j < lastLayerLength; j++) {
                    const error = actualOutput[j] - expectedOutput[j];
                    errors[lastLayerOffset + j] = error * this._sigmoidDerivative(actualOutput[j]);
                }

                for (let l = this.layerLengths.length - 2; l > 0; l--) {
                    const currentLayerLength = this.layerLengths[l];
                    const nextLayerLength = this.layerLengths[l + 1];
                    const currentLayerOffset = this.layerOffsets[l];
                    const nextLayerOffset = this.layerOffsets[l + 1];

                    for (let j = 0; j < currentLayerLength; j++) {
                        const currentNeuronIndex = currentLayerOffset + j;
                        let sum = 0;
                        for (let n = 0; n < nextLayerLength; n++) {
                            sum += errors[nextLayerOffset + n] * this.weights[this.weightOffsets[l] + j + n * currentLayerLength];
                        }
                        errors[currentNeuronIndex] = sum * this._reluDerivative(this.layerSums[currentNeuronIndex]);
                    }
                }

                let weightIndex = 0;
                let biasIndex = 0;
                for (let l = 0; l < this.layerLengths.length - 1; l++) {
                    const previousLayerLength = this.layerLengths[l];
                    const previousLayerOffset = this.layerOffsets[l];
                    const currentLayerLength = this.layerLengths[l + 1];
                    const currentLayerOffset = this.layerOffsets[l + 1];

                    for (let j = 0; j < currentLayerLength; j++) {
                        const error = errors[currentLayerOffset + j];
                        biasGrad[biasIndex] += error;
                        biasIndex++;

                        for (let i = 0; i < previousLayerLength; i++) {
                            weightGrad[weightIndex] += error * this.layerValues[previousLayerOffset + i];
                            weightIndex++;
                        }
                    }
                }
            }

            let lerp = 0.9;
            let epsilon = 1e-8;



            for (let i = 0; i < this.weights.length; i++) {
                let momentum = weightGrad[i] / inputs.length;
                this.weightVelocities[i] = lerp * this.weightVelocities[i] + (1 - lerp) * momentum * momentum;
                this.weights[i] += -learningRate / Math.sqrt(this.weightVelocities[i] + epsilon) * momentum;
            }
            for (let i = 0; i < this.biases.length; i++) {
                let momentum = biasGrad[i] / inputs.length;
                this.biasVelocities[i] = lerp * this.biasVelocities[i] + (1 - lerp) * momentum * momentum;
                this.biases[i] += -learningRate / Math.sqrt(this.biasVelocities[i] + epsilon) * momentum;
            }

        }

        return loss / inputs.length;
    }

    clone() {
        const clone = new NN(this.layerLengths);
        clone.loadChromosome(this.getChromosome());
        return clone;
    }
}
