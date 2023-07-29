
def get_args():
  import argparse

  parser = argparse.ArgumentParser(description='Squeezenet model for TF2')

  parser.add_argument('-m', '--mode', type=str, default='train', help='Can be "train" or "test"')
  parser.add_argument('-d','--checkpoint_dir', type=str, default='./', help='Directory to store checkpoints during training')
  parser.add_argument('-r', '--restore_checkpoint', action='store_true', help='Use this flag if you want to resume training from a previous checkpoint')
  parser.add_argument('-b', '--batch_size', type=int, default=512, help='Number of images per batch fed through network')
  parser.add_argument('-e', '--num_epochs', type=int, default=100, help='Number of passes through training data before stopping')

  return parser.parse_args()
