from keras.callbacks import Callback




class ReviewCheckpoint(Callback):

    def __init__(self, X_test, y_test, args ):
        super(ReviewCheckpoint, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.args = args



    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import log_loss

        best = log_loss(self.y_test, self.model.predict_proba(self.X_test))

        print(f'{"="*5}The Score on validate set is: {round(best,4)}, epoch:{epoch+1}, with: {self.args}')

        epoch += 1



if __name__ == '__main__':
    pass
    #df_test  = get_test()
    # model = Inception_tranfer(-1, unlock=0).gen_model()
    # show_img_with_tags(df_test, -1, 224, model, random=False)


