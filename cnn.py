import tensorflow as tf
from tf.estimator import Estimator, EstimatorSpec

'''
https://www.tensorflow.org/tutorials/layers
https://www.tensorflow.org/get_started/custom_estimators
https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
'''

class CNNClassifier(Estimator):
    def __init__(self,
                 feature_columns,
                 convolutional_layers,
                 pooling_layers,
                 dense_layers,
                 n_classes,
                 optimizer,
                 model_dir=None, config=None, warm_start_from=None):
        '''
        Example
        feature_columns = [feature_a, feature_b]

        convolutional_layers = [
            {'filters': 32,
            'kernel_size': [5, 5], # allows integer for x=y
            'padding': 'same',     # defaulted to valid
            'activation': 'relu'}, # TODO: this is now always relu
            {}
            ]
        pooling_layers = [
            {'pool_size': [2, 2],   # allows integer for x=y
            'strides': 2},
            {}
            ]
        assert(len(pooling_layers) == len(convolutional_layers))
        dense_layers = [
            {'units': 1024,
            'activation': 'relu', # TODO: this is now always relu
            'dropout': 0.4},      # defaulted to None
            {}
            ]
        n_classes = 10
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        '''
        # Sanity checks
        if len(pooling_layers) != len(convolutional_layers):
            raise RuntimeError('len(pooling_layers) != len(convolutional_layers)')
        
        # Make it params
        params = {
            'feature_columns': feature_columns,
            'convolutional_layers': convolutional_layers,
            'pooling_layers': pooling_layers,
            'dense_layers': dense_layers,
            'logit_layers': logit_layers,
            'n_classes': n_classes
            }
        
        super().__init__(model_fn=self._model_fn,
                         model_dir=model_dir,
                         config=config,
                         params=params,
                         warm_start_from=warm_start_from)
        
    def _construct_network(input_layer, params):
        '''
        Returns:
        logits: last layer before applying softmax to calculate loss
        '''
        net = input_layer
        
        # Convolutional + pooling
        convolutional_layers = params['convolutional_layers']
        pooling_layers = params['pooling_layers']
        
        for conv, pool in zip(convolutional_layers, pooling_layers):
            # Convolutional layer
            padding = 'valid'
            if 'padding' in conv:
                padding = conv['padding']
            net = tf.layers.conv2d(inputs=net,
                                   filters=conv['filters'],
                                   kernel_size=conv['kernel_size'],
                                   padding=padding,
                                   activation=tf.nn.relu)
            # Pooling layer
            net = tf.layers.max_pooling2d(inputs=net,
                                          pool_size=pool['pool_size'],
                                          strides=pool['strides'])
        
        # TODO: Flatten, need to get original image sizes
        # and resize according to convolutional and pooling layers
        net = tf.reshape(net, [-1, 7 * 7 * 64])
        
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


        
        
    def _model_fn(features, labels, mode, params=None):
        '''
        features shape is: [batch_size, image_height, image_width, channels]
        '''
        input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
        logits = _construct_network(input_layer, params)
        
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
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions["classes"])
            }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
