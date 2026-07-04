import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        # Ensure step is a float32 tensor
        step = tf.cast(step, tf.float32)

        # Handle step = 0 to avoid division by zero
        # Add a small epsilon to step to prevent rsqrt(0) or step * (0)^-1.5 errors
        arg1 = tf.math.rsqrt(step + 1e-9) # Add 1e-9 to step to avoid rsqrt(0)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
