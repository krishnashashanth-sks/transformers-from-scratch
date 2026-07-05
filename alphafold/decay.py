import tensorflow as tf

# --- Optimizers and Learning Rate Schedules ---
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        warmup_steps,
        alpha=0.0,
        name=None
    ):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            step = tf.cast(step, tf.float32)

            warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)

            global_step_val = tf.maximum(0.0, step - self.warmup_steps)
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.1415926535) * global_step_val / self.decay_steps))
            decayed_lr = self.initial_learning_rate * cosine_decay + self.alpha

            learning_rate = tf.where(
                step < self.warmup_steps,
                warmup_lr,
                decayed_lr
            )
            return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate.numpy(),
            "decay_steps": self.decay_steps.numpy(),
            "warmup_steps": self.warmup_steps.numpy(),
            "alpha": self.alpha.numpy(),
            "name": self.name,
        }
