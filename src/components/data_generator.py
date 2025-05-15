import numpy as np
import tensorflow as tf

class SatelliteDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=8, shuffle=True, prefetch_factor=2):
        """
        Data generator for memory-efficient training with prefetching
        Args:
            X: Input data of shape (samples, time_steps, height, width)
            y: Target data of shape (samples, time_steps, height, width)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data at each epoch
            prefetch_factor: Number of batches to prefetch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.indexes = np.arange(len(self.X))
        
        # Initialize prefetch queue
        self.prefetch_queue = []
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        # Prefetch initial batches
        self._prefetch_batches()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def _prefetch_batches(self):
        """Prefetch next batches"""
        while len(self.prefetch_queue) < self.prefetch_factor:
            idx = len(self.prefetch_queue)
            if idx >= self.__len__():
                break
                
            # Get batch indexes
            batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
            
            # Get batch data
            batch_X = self.X[batch_indexes]
            batch_y = self.y[batch_indexes]
            
            # Add channel dimension if needed
            if len(batch_X.shape) == 4:  # (batch, time, height, width)
                batch_X = np.expand_dims(batch_X, axis=-1)
            if len(batch_y.shape) == 4:
                batch_y = np.expand_dims(batch_y, axis=-1)
            
            # Convert to TensorFlow tensors for faster GPU transfer
            batch_X = tf.convert_to_tensor(batch_X, dtype=tf.float32)
            batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
            
            self.prefetch_queue.append((batch_X, batch_y))
    
    def __getitem__(self, idx):
        """Get batch at position idx with prefetching"""
        # If we're getting close to the end of our prefetch queue, fetch more
        if idx >= len(self.prefetch_queue) - self.prefetch_factor // 2:
            self._prefetch_batches()
        
        # Get and remove the first batch from the queue
        batch = self.prefetch_queue.pop(0)
        
        # Prefetch new batch
        self._prefetch_batches()
        
        return batch
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            # Clear and refill prefetch queue with new shuffled data
            self.prefetch_queue.clear()
            self._prefetch_batches() 