import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.training.common_utils import onehot

from src.dog import DoGJAX, LDoGJAX, polynomial_decay_averaging, get_av_model

import tensorflow_datasets as tfds


def train_epoch(state, train_loader, train_step, epoch, log_interval):
    """Train the model for one epoch."""
    metrics = []
    for i, batch in enumerate(train_loader(), start=1):
        batch = jax.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), batch)
        state, loss = train_step(state, (batch['image'], batch['label']))
        metrics.append(loss)
        if i % log_interval == 0:
            print(f'Epoch: {epoch}, Step: {i}, Loss: {np.mean(metrics)}')
            metrics = []
    return state, {'loss': np.mean(metrics)}


def compute_loss(logits, labels):
    """Compute the loss."""
    return -jnp.mean(jnp.sum(labels * logits, axis=-1))


def compute_accuracy(logits, labels):
    """Compute the accuracy."""
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))


def test_epoch(state, test_loader):
    """Test the model."""
    test_loss = []
    test_accuracy = []
    test_loss_av = []
    test_accuracy_av = []
    for batch in test_loader():
        batch = jax.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), batch)
        labels = onehot(batch['label'], 10)

        logits = state.apply_fn({'params': state.params}, batch['image'])
        test_loss.append(compute_loss(logits, labels))
        test_accuracy.append(compute_accuracy(logits, labels))

        logits = state.apply_fn({'params': get_av_model(state.opt_state)}, batch['image'])
        test_loss_av.append(compute_loss(logits, labels))
        test_accuracy_av.append(compute_accuracy(logits, labels))
    return {'loss': np.mean(jnp.array(test_loss)),
            'accuracy': np.mean(jnp.array(test_accuracy)),
            'loss_av': np.mean(jnp.array(test_loss_av)),
            'accuracy_av': np.mean(jnp.array(test_accuracy_av))}


def get_data_loader(data_dir, batch_size, split='train'):
    """Get data loader."""
    dataset_builder = tfds.builder('mnist', data_dir=data_dir)
    dataset_builder.download_and_prepare()
    ds = dataset_builder.as_dataset(split=split)
    ds = ds.batch(batch_size)

    def gen():
        for batch in tfds.as_numpy(ds):
            images = batch['image'] / 255.0  # normalization
            labels = batch['label']
            yield {'image': images, 'label': labels}

    return gen


def get_data_loaders(train_batch_size, test_batch_size, data_dir):
    """Get train and test data loaders."""
    train_ds = get_data_loader(data_dir=data_dir, batch_size=train_batch_size, split='train')
    test_ds = get_data_loader(data_dir=data_dir, batch_size=test_batch_size, split='test')
    return train_ds, test_ds


class Net(nn.Module):
    """Define the neural network architecture."""

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=10)

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jnp.reshape(x, (x.shape[0], -1))  # equivalent of torch.flatten(x, 1)
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return jax.nn.log_softmax(x, axis=-1)


@jax.jit
def train_step(state, batch):
    """Define a training step."""

    def loss_fn(params):
        inputs, targets = batch
        preds = state.apply_fn({'params': params}, inputs)
        return -jnp.mean(jnp.sum(onehot(targets, 10) * preds, axis=-1))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)

    return new_state, loss


def create_model_and_optimizer(ldog, lr, reps_rel, gamma, weight_decay, seed=0, init_eta=None):
    """Create model and optimizer."""
    model = Net()

    init_rng = jax.random.PRNGKey(seed)
    inputs = jnp.ones([1, 28, 28, 1], jnp.float32)
    initial_params = model.init(init_rng, inputs)['params']

    opt_class = LDoGJAX if ldog else DoGJAX
    optimizer = opt_class(learning_rate=lr, reps_rel=reps_rel, eps=1e-8, init_eta=init_eta, weight_decay=weight_decay)

    averager = polynomial_decay_averaging(gamma=gamma)
    optimizer = optax.chain(optimizer, averager)

    return initial_params, model, optimizer


def main():
    """Main function."""
    # Training settings
    data_dir = '/specific/netapp5/joberant/data_nobck/maorivgi/cache/data'
    batch_size = 64
    test_batch_size = 1000
    epochs = 14
    ldog = True
    lr = 1.0
    reps_rel = 1e-6
    init_eta = None
    avg_gamma = 8
    weight_decay = 0
    seed = 1
    log_interval = 10
    save_model = False

    train_loader, test_loader = get_data_loaders(batch_size, test_batch_size, data_dir)

    initial_params, model, optimizer = create_model_and_optimizer(ldog, lr, reps_rel, avg_gamma, weight_decay, seed,
                                                                  init_eta)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=optimizer)

    for epoch in range(1, epochs + 1):
        state, train_metrics = train_epoch(state, train_loader, train_step, epoch, log_interval)
        test_metrics = test_epoch(state, test_loader)

        print(f'Test set: Loss = {test_metrics["loss"]:.4f}, '
              f'Accuracy = {test_metrics["accuracy"] * 100:.2f}%, '
              f'Loss (avg) = {test_metrics["loss_av"]:.4f}, '
              f'Accuracy (avg) = {test_metrics["accuracy_av"] * 100:.2f}%')

    if save_model:
        checkpoints.save_checkpoint('.', state, epoch)

    print("Finished Training!")

if __name__ == '__main__':
    main()

