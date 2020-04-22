import tensorflow as tf
import warnings  
import tqdm

tf.compat.v1.enable_eager_execution()

class Transliteration():

	def __init__(self,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,num_decoder_tokens,hidden_units):
		self.max_encoder_seq_length = max_encoder_seq_length
		self.max_decoder_seq_length = max_decoder_seq_length
		self.num_encoder_tokens = num_encoder_tokens
		self.num_decoder_tokens = num_decoder_tokens
		self.hidden_units = hidden_units

	def create_model(self):

		encoder_inputs = tf.keras.layers.Input(shape=(None,self.num_encoder_tokens))
		encoder = tf.keras.layers.LSTM(self.hidden_units, return_state=True)
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		encoder_states = [state_h, state_c]

		decoder_inputs = tf.keras.layers.Input(shape=(None,self.num_decoder_tokens))
		decoder_lstm = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
		decoder_dense = tf.keras.layers.Dense(self.num_decoder_tokens, activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)

		self.model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
		
		return self.model

	def inference_model(self,model):

		encoder_inputs = model.input[0] #input_1
		encoder_outputs, state_h, state_c = model.layers[2].output # lstm_1
		encoder_states = [state_h, state_c]
		self.encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

		decoder_inputs = model.input[1] #input_2
		decoder_state_input_h = Input(shape=(self.hidden_units,),name='input_3')
		decoder_state_input_c = Input(shape=(self.hidden_units,),name='input_4')
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		decoder_lstm = model.layers[3]	#lstm_2
		decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
		decoder_states = [state_h_dec, state_c_dec]
		decoder_dense = model.layers[4] #dense_1
		decoder_outputs=decoder_dense(decoder_outputs)

		self.decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

		return self.encoder_model, self.decoder_model

	def compile_model(self,learning_rate):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


	def loss(self,encoder_input_data,decoder_input_data,decoder_target_data, training):
		y_ = self.model([encoder_input_data,decoder_input_data],decoder_target_data)
		return self.loss_object(y_true=decoder_target_data, y_pred=y_)

	def grad(self,encoder_input_data,decoder_input_data,decoder_target_data):
		with tf.GradientTape() as tape:
			loss_value = self.loss(encoder_input_data,decoder_input_data,decoder_target_data,training=True)
		return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

	def fit(self,epochs,batch_size,encoder_input_data,decoder_input_data,decoder_target_data,path):
		
		total_batch = int(len(encoder_input_data)/batch_size)

		for epoch in range(epochs):
			self.train_loss_results = []
			self.train_accuracy_results = []

			epoch_loss_avg = tf.keras.metrics.Mean()
			epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

			start = 0
			end = batch_size
			for i in tqdm.tqdm(range(total_batch)):
				batch_encoder_input = tf.keras.backend.one_hot(encoder_input_data[start:end],self.num_encoder_tokens)
				batch_decoder_input = tf.keras.backend.one_hot(decoder_target_data[start:end],self.num_decoder_tokens)
				batch_decoder_target = tf.keras.backend.one_hot(decoder_target_data[start:end],self.num_decoder_tokens)
				# Optimize the model
				loss_value, grads = self.grad(batch_encoder_input,batch_decoder_input,batch_decoder_target)
				self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
				# Track progress
				epoch_loss_avg.update_state(loss_value)
				epoch_accuracy.update_state(batch_decoder_target, self.model([batch_encoder_input,batch_decoder_input], training=True))
				start = start + batch_size
				end = end + batch_size

			# End epoch
			if epoch % 5 == 0:
				#print('Saving Weights....')
				name = (path+'weights%04d.hdf5') % epoch
				self.model.save_weights(name)

			self.train_loss_results.append(epoch_loss_avg.result())
			self.train_accuracy_results.append(epoch_accuracy.result())

			print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result().numpy(),epoch_accuracy.result().numpy()))

		return self.train_loss_results,self.train_accuracy_results


	def save(self,path):
		self.model.save(path)


