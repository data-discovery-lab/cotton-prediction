import tensorflow as tf


class GAN:

    """
    A GAN model.
    :param real_size: The shape of the real data.
    :param z_size: The number of entries in the z code vector.
    :param learnin_rate: The learning rate to use for Adam.
    :param num_classes: The number of classes to recognize.
    :param alpha: The slope of the left half of the leaky ReLU activation
    :param beta1: The beta1 parameter for Adam.
    """
    def __init__(self, real_size, z_size, learning_rate, num_classes=10, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()

        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        inputs = self.model_inputs(real_size, z_size)
        self.input_real, self.input_z, self.y, self.label_mask = inputs
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")

        loss_results = self.model_loss(self.input_real, self.input_z,
                                  real_size[2], self.y, num_classes,
                                  label_mask=self.label_mask,
                                  alpha=0.2,
                                  drop_rate=self.drop_rate)
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples = loss_results

        self.d_opt, self.g_opt, self.shrink_lr = self.model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1)

    def model_inputs(self, real_dim, z_dim):
        # placeholder for inputing the real images from the training set to the discriminator
        inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')

        # placeholder for inputing random noise data into the discriminator
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

        # placeholder for inputing the real data labels (0-9 for the SVHN dataset)
        y = tf.placeholder(tf.int32, (None), name='y')

        # placeholder for inputing the label masks which tell the model for which
        # sample images should it take the labels for training
        label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

        return inputs_real, inputs_z, y, label_mask

    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """
        # Get weights and biases to update. Get them separately for the discriminator and the generator
        discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # Minimize both players' costs simultaneously
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='d_optimizer').minimize(d_loss,
                                                                                                var_list=discriminator_train_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='g_optimizer').minimize(g_loss,
                                                                                                var_list=generator_train_vars)

        shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

        return d_train_opt, g_train_opt, shrink_lr

    def model_loss(self, input_real, input_z, output_dim, y, num_classes, label_mask, alpha=0.2, drop_rate=0., smooth=0.1):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input random noise vector
        :param output_dim: The number of channels in the output image
        :param y: Integer class labels
        :param num_classes: The number of classes
        :param alpha: The slope of the left half of leaky ReLU activation
        :param drop_rate: The probability of dropping a hidden unit
        :return: A tuple of (discriminator loss, generator loss)
        """

        # These numbers multiply the size of each layer of the generator and the discriminator,
        # respectively. You can reduce them to run your code faster for debugging purposes.
        g_size_mult = 32
        d_size_mult = 64

        # Here we instatiate the generator and the discriminator networks
        g_model = self.generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
        d_on_data = self.discriminator(input_real, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)

        # d_model_real: probability that the input is real
        # class_logits_on_data: the unnormalized log probability values for the probability of each classe
        # gan_logits_on_data: the probability of whether or not the image is real
        # data_features: features from the last layer of the discriminator to be used in the feature matching loss
        d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data

        d_on_samples = self.discriminator(g_model, reuse=True, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
        d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples

        # Here we compute `d_loss`, the loss for the discriminator.
        # This should combine two different losses:
        # 1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
        #    real-vs-fake classification problem.
        real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                                                labels=tf.ones_like(gan_logits_on_data) * (
                                                                                1 - smooth)))

        fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                                                labels=tf.zeros_like(
                                                                                    gan_logits_on_samples)))

        unsupervised_loss = real_data_loss + fake_data_loss

        #  2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
        #     for the multi-class softmax. For this one we use the labels. Don't forget to ignore
        #     use `label_mask` to ignore the examples that we are pretending are unlabeled for the
        #     semi-supervised learning problem.
        y = tf.squeeze(y)
        suppervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                                   labels=tf.one_hot(y, num_classes, dtype=tf.float32))

        label_mask = tf.squeeze(tf.to_float(label_mask))

        # ignore the labels that we pretend does not exist for the loss
        suppervised_loss = tf.reduce_sum(tf.multiply(suppervised_loss, label_mask))

        # get the mean
        suppervised_loss = suppervised_loss / tf.maximum(1.0, tf.reduce_sum(label_mask))
        d_loss = unsupervised_loss + suppervised_loss

        # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
        # This loss consists of minimizing the absolute difference between the expected features
        # on the data and the expected features on the generated samples.
        # This loss works better for semi-supervised learning than the tradition GAN losses.

        # Make the Generator output features that are on average similar to the features
        # that are found by applying the real data to the discriminator

        data_moments = tf.reduce_mean(data_features, axis=0)
        sample_moments = tf.reduce_mean(sample_features, axis=0)
        g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

        pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
        eq = tf.equal(tf.squeeze(y), pred_class)
        correct = tf.reduce_sum(tf.to_float(eq))
        masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

        return d_loss, g_loss, correct, masked_correct, g_model

    def generator(self, z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
        with tf.variable_scope('generator', reuse=reuse):
            # First fully connected layer
            x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
            # Reshape it to start the convolutional stack
            x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
            x1 = tf.layers.batch_normalization(x1, training=training)
            x1 = tf.maximum(alpha * x1, x1)

            x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2, training=training)
            x2 = tf.maximum(alpha * x2, x2)

            x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=training)
            x3 = tf.maximum(alpha * x3, x3)

            # Output layer
            logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')

            out = tf.tanh(logits)

            return out

    def discriminator(self, x, reuse=False, alpha=0.2, drop_rate=0., num_classes=10, size_mult=64):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.dropout(x, rate=drop_rate / 2.5)

            # Input layer is ?x32x32x3
            x1 = tf.layers.conv2d(x, size_mult, 3, strides=2, padding='same')
            relu1 = tf.maximum(alpha * x1, x1)
            relu1 = tf.layers.dropout(relu1, rate=drop_rate)  # [?x16x16x?]

            x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
            bn2 = tf.layers.batch_normalization(x2, training=True)  # [?x8x8x?]
            relu2 = tf.maximum(alpha * bn2, bn2)

            x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same')  # [?x4x4x?]
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(alpha * bn3, bn3)
            relu3 = tf.layers.dropout(relu3, rate=drop_rate)

            x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
            bn4 = tf.layers.batch_normalization(x4, training=True)
            relu4 = tf.maximum(alpha * bn4, bn4)

            x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
            bn5 = tf.layers.batch_normalization(x5, training=True)
            relu5 = tf.maximum(alpha * bn5, bn5)

            x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same')  # [?x2x2x?]
            bn6 = tf.layers.batch_normalization(x6, training=True)
            relu6 = tf.maximum(alpha * bn6, bn6)
            relu6 = tf.layers.dropout(relu6, rate=drop_rate)

            x7 = tf.layers.conv2d(relu5, filters=(2 * size_mult), kernel_size=3, strides=1, padding='valid')
            # Don't use bn on this layer, because bn would set the mean of each feature
            # to the bn mu parameter.
            # This layer is used for the feature matching loss, which only works if
            # the means can be different when the discriminator is run on the data than
            # when the discriminator is run on the generator samples.
            relu7 = tf.maximum(alpha * x7, x7)

            # Flatten it by global average pooling
            # In global average pooling, for every feature map we take the average over all the spatial
            # domain and return a single value
            # In: [BATCH_SIZE,HEIGHT X WIDTH X CHANNELS] --> [BATCH_SIZE, CHANNELS]
            features = tf.reduce_mean(relu7, axis=[1, 2])

            # Set class_logits to be the inputs to a softmax distribution over the different classes
            class_logits = tf.layers.dense(features, num_classes)

            # This function is more numerically stable than log(sum(exp(input))).
            # It avoids overflows caused by taking the exp of large inputs and underflows
            # caused by taking the log of small inputs.
            gan_logits = tf.reduce_logsumexp(class_logits, 1)

            # Get the probability that the input is real rather than fake
            out = tf.nn.softmax(class_logits)  # class probabilities for the 10 real classes plus the fake class

            return out, class_logits, gan_logits, features

