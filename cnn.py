import tensorflow as tf

'''
https://www.tensorflow.org/tutorials/layers
https://www.tensorflow.org/get_started/custom_estimators
https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
'''

class CNNClassifier(tf.estimator.Estimator):
    def __init__(self,
                 images_shape,
                 convolutional_layers,
                 pooling_layers,
                 dense_layers,
                 n_classes,
                 optimizer,
                 model_dir=None, config=None, warm_start_from=None):
        '''
        Example:
        images_shape = [width, height, channels]
        convolutional_layers = [
            {'filters': 32,
            'kernel_size': [5, 5], # allows integer for x=y
            'padding': 'same',     # defaulted to valid
            'activation': 'relu'}, # TODO: this is now always relu
            {etc}
            ]
        pooling_layers = [
            {'pool_size': [2, 2],   # allows integer for x=y
            'strides': 2},
            {etc}
            ]
        assert(len(pooling_layers) == len(convolutional_layers))
        dense_layers = [
            {'units': 1024,
            'activation': 'relu', # TODO: this is now always relu
            'dropout': 0.4},      # defaulted to None
            {etc}
            ]
        n_classes = 10
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        '''
        # Sanity checks (most already done by tf)
        if len(images_shape) != 3:
            raise RuntimeError('images_shape must be in the shape of [width, height, channels]')
        if len(pooling_layers) != len(convolutional_layers):
            raise RuntimeError('len(pooling_layers) != len(convolutional_layers)')
        
        # Make it params
        params = {
            'images_shape': images_shape,
            'convolutional_layers': convolutional_layers,
            'pooling_layers': pooling_layers,
            'dense_layers': dense_layers,
            'n_classes': n_classes,
            'optimizer': optimizer
            }
        
        super().__init__(model_fn=self._model_fn,
                         model_dir=model_dir,
                         config=config,
                         params=params,
                         warm_start_from=warm_start_from)
        
    def _construct_network(self, input_layer, mode, params):
        '''
        Returns:
        logits: last layer before applying softmax to calculate loss
        '''
        #### print("Input layer shape:", input_layer.shape)
        width  = params['images_shape'][0]
        height = params['images_shape'][1]
        depth  = params['images_shape'][2]
        net = tf.reshape(input_layer, [-1, width, height, depth])
        #### print("Reshaped input layer shape:", net.shape)
        # Convolutional + pooling
        convolutional_layers = params['convolutional_layers']
        pooling_layers = params['pooling_layers']
        
        for conv, pool in zip(convolutional_layers, pooling_layers):
            # Convolutional layer
            # Pre-process
            padding = 'valid'
            if 'padding' in conv:
                padding = conv['padding']
            k_size = conv['kernel_size']
            if isinstance(k_size, int):
                k_size = [k_size, k_size]
            
            net = tf.layers.conv2d(inputs=net,
                                   filters=conv['filters'],
                                   kernel_size=k_size,
                                   padding=padding,
                                   activation=tf.nn.relu)
            depth = conv['filters'] # Depth is equal to the last number of filters/channels
            if padding == 'valid':
                width = width - k_size[0] + 1
                height = height  - k_size[1] + 1
            # Pooling layer
            # Pre-process
            p_size = pool['pool_size']
            if isinstance(p_size, int):
                p_size = [p_size, p_size]
            strides = pool['strides']
            if isinstance(strides, int):
                strides = [strides, strides]
            net = tf.layers.max_pooling2d(inputs=net,
                                          pool_size=p_size,
                                          strides=strides)
            width  = (width - p_size[0])//strides[0] + 1
            height = (height - p_size[1])//strides[1] + 1

        #### print("Logits input layer shape:", net.shape)
        #### print("width: {}\nheight: {}\ndepth: {}".format(width, height, depth))
        net = tf.reshape(net, [-1, width * height * depth])
        #### print("Logits output layer shape:", net.shape)
        
        # Dense layers
        dense_layers = params['dense_layers']
        for dense in dense_layers:
            net = tf.layers.dense(inputs=net, units=dense['units'], activation=tf.nn.relu)
            if 'dropout' in dense:
                net = tf.layers.dropout(inputs=net,
                                        rate=dense['dropout'],
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))
        
        # Logits layer
        logits = tf.layers.dense(inputs=net, units=params['n_classes'])
        return logits
 
    def _model_fn(self, features, labels, mode, params):
        '''
        features shape is: [batch_size, image_height, image_width, channels]
        '''
        logits = self._construct_network(features, mode, params)
        
        # TODO: READ AND REWRITE THIS
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params['optimizer']
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
