

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

import os

from lib import MovieReviewClassificationModel as mrcm

from lib.args import get_args

args = get_args()

def main():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    model = mrcm.MovieReviewClassificationModel()


    ckpt_dir = os.path.join(args.checkpoint_dir, "checkpointsMy/")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.restore_checkpoint or args.mode == 'test':
        print("-------------------------- Restoring checkpoint")
        model.load_weights(ckpt_dir)
    
    model.compile_model()

    if args.mode == 'train':

      model.train(ckpt_dir)

      results = model.evaluate()
      print(results)

      print("-------------------------- Graph of accuracy over time")

      history_dict = model.history.history
      history_dict.keys()

      draw_charts(history_dict)

    if args.mode == 'predict':
        # print("-------------------------- Loading weights")
        # model.load_weights('./checkpoints/my_checkpoint')
        prediction = model.predict(["This is a worst movie ever!", "This is a bad movie!", "This is an average movie!", "This is an amazing movie! The best ever!"])
        print("Prediction:", prediction)


def draw_charts(history_dict):
    
      acc = history_dict['accuracy']
      print("Accuracy:", acc)
      val_acc = history_dict['val_accuracy']
      loss = history_dict['loss']
      val_loss = history_dict['val_loss']

      epochs = range(1, len(acc) + 1)

      # "bo" is for "blue dot"
      plt.plot(epochs, loss, 'bo', label='Training loss')
      # b is for "solid blue line"
      plt.plot(epochs, val_loss, 'b', label='Validation loss')
      plt.title('Training and validation loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()

      # plt.show(block=True)
      plt.savefig('loss.png')


      plt.clf()   # clear figure

      plt.plot(epochs, acc, 'bo', label='Training acc')
      plt.plot(epochs, val_acc, 'b', label='Validation acc')
      plt.title('Training and validation accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()

      # plt.show()
      plt.savefig('accuracy.png')


if __name__ == '__main__':
    main()

