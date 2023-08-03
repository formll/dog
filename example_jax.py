# pip install jax flax tensorflow-datasets
# conda create -n jdog python=3.8
# conda install -c conda-forge jax=0.3.25
# pip install --upgrade jaxlib==0.4.11
# conda install -c conda-forge flax
# conda install -c conda-forge tensorflow tensorflow-datasets
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from flax.training.common_utils import onehot
from flax.training import checkpoints
from src.jax.dog import DoG  # , LDoG, PolynomialDecayAverager
import tensorflow_datasets as tfds

def train_epoch(state, train_loader, train_step, epoch, log_interval):
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
    return -jnp.mean(jnp.sum(labels * logits, axis=-1))


def compute_accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))


def test_epoch(state, test_loader):
    test_loss = []
    test_accuracy = []
    for batch in test_loader():
        batch = jax.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), batch)
        labels = onehot(batch['label'], 10)
        logits = state.apply_fn({'params': state.params}, batch['image'])
        test_loss.append(compute_loss(logits, labels))
        test_accuracy.append(compute_accuracy(logits, labels))
    return {'loss': np.mean(jnp.array(test_loss)),
            'accuracy': np.mean(jnp.array(test_accuracy))}




def get_data_loader(data_dir, batch_size, split='train'):
    dataset_builder = tfds.builder('mnist', data_dir=data_dir)
    dataset_builder.download_and_prepare()
    ds = dataset_builder.as_dataset(split=split)
    ds = ds.batch(batch_size)

    def gen():
        for batch in tfds.as_numpy(ds):
            images = batch['image'] / 255.0  # normalization
            # images = np.expand_dims(images, axis=-1)
            labels = batch['label']
            yield {'image': images, 'label': labels}

    return gen


def get_data_loaders(train_batch_size, test_batch_size, data_dir):
    train_ds = get_data_loader(data_dir=data_dir, batch_size=train_batch_size, split='train')
    test_ds = get_data_loader(data_dir=data_dir, batch_size=test_batch_size, split='test')
    return train_ds, test_ds


# TODO - something about the network doesn't work as expected. Need to start with SGD, see that it works and then move to dog
class Net(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=10)
        self.dropout1 = nn.Dropout(rate=0.25)
        self.dropout2 = nn.Dropout(rate=0.5)

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = self.dropout1(x)  # couldn't make it work due to something with "detemenistic" needing to be passed
        x = jnp.reshape(x, (x.shape[0], -1))  # equivalent of torch.flatten(x, 1)
        x = self.fc1(x)
        x = jax.nn.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return jax.nn.log_softmax(x, axis=-1)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        inputs, targets = batch
        preds = state.apply_fn({'params': params}, inputs)
        return -jnp.mean(jnp.sum(onehot(targets, 10) * preds, axis=-1))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)

    return new_state, loss


def create_model_and_optimizer(ldog, lr, seed=0):
    model = Net()

    init_rng = jax.random.PRNGKey(seed)
    inputs = jnp.ones([1, 28, 28, 1], jnp.float32)
    initial_params = model.init(init_rng, inputs)['params']

    # opt_class = init_ldog_params if ldog else init_dog_params
    opt_class = DoG
    optimizer = opt_class(learning_rate=lr, reps_rel=1e-6, eps=1e-8, init_eta=None, weight_decay=0.0)

    # import optax
    # optimizer = optax.adam(learning_rate=lr)

    return initial_params, model, optimizer


def main():
    # Training settings
    data_dir = '/specific/netapp5/joberant/data_nobck/maorivgi/cache/data'
    batch_size = 64
    test_batch_size = 1000
    epochs = 14
    ldog = False
    lr = 1.0
    reps_rel = 1e-6
    init_eta = 0
    avg_gamma = 8
    weight_decay = 0
    seed = 1
    log_interval = 10
    save_model = False
    log_state = True
    use_cuda = False #jax.devices("gpu")

    device = use_cuda[0] if use_cuda else jax.devices("cpu")[0]

    # here we will have a placeholder function to simulate PyTorch's DataLoader
    train_loader, test_loader = get_data_loaders(batch_size, test_batch_size, data_dir)

    initial_params, model, optimizer = create_model_and_optimizer(ldog, lr, seed)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=optimizer)

    # averager = PolynomialDecayAverager(model, gamma=avg_gamma)

    for epoch in range(1, epochs + 1):
        state, train_metrics = train_epoch(state, train_loader, train_step, epoch, log_interval)
        test_metrics = test_epoch(state, test_loader)
        #
        print('Test set: Loss = {:.4f}, Accuracy = {:.2f}%\n'.format(
            test_metrics['loss'], test_metrics['accuracy'] * 100))
        print('hi')
        pass

    if save_model:
        checkpoints.save_checkpoint(".", state, epoch, keep=3)
        # checkpoints.save_checkpoint(".", averager.state, epoch, keep=3, prefix="averaged_model_")


if __name__ == '__main__':
    main()
