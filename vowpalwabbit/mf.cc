/*
 Copyright (c) by respective owners including Yahoo!, Microsoft, and
 individual contributors. All rights reserved.  Released under a BSD (revised)
 license as described in the file LICENSE.
 */
#include <fstream>
#include <float.h>
#ifdef _WIN32
#include <winsock2.h>
#else
#include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <map>
#include "parse_example.h"
#include "constant.h"
#include "sparse_dense.h"
#include "gd.h"
#include "cache.h"
#include "simple_label.h"
#include "rand48.h"
#include "vw.h"
#include <algorithm>
#include "hash.h"
#include <sstream>
#include "parse_primitives.h"

using namespace std;

namespace MF {

struct mf {
	learner base;
	vector<string> pairs;

	double lin_contraction, left_contraction, right_contraction;

	vw* all;
};



void save_load(void* d, io_buf& model_file, bool read, bool text) {
	vw* all = (vw*) d;
	uint32_t length = 1 << all->num_bits;
	uint32_t stride = all->reg.stride;

	if (read) {
		initialize_regressor(*all);
		if (all->random_weights)
			for (size_t j = 0; j < all->reg.stride * length; j++)
				all->reg.weight_vector[j] = (float) (0.1 * frand48());
	}

	if (model_file.files.size() > 0) {
		uint32_t i = 0;
		uint32_t text_len;
		char buff[512];
		size_t brw = 1;

		do {
			brw = 0;
			size_t K = all->rank * 2 + 1;

			text_len = sprintf(buff, "%d ", i);
			brw += bin_text_read_write_fixed(model_file, (char *) &i, sizeof(i),
					"", read, buff, text_len, text);
			if (brw != 0)
				for (uint32_t k = 0; k < K; k++) {
					uint32_t ndx = stride * i + k;

					weight* v = &(all->reg.weight_vector[ndx]);
					text_len = sprintf(buff, "%f ", *v);
					brw += bin_text_read_write_fixed(model_file, (char *) v,
							sizeof(*v), "", read, buff, text_len, text);

				}
			if (text)
				brw += bin_text_read_write_fixed(model_file, buff, 0, "", read,
						"\n", 1, text);

			if (!read)
				i++;
		} while ((!read && i < length) || (read && brw > 0));
	}
}

float inline_predict(mf* data, vw* all, example* &ec, unsigned char temp_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	float quad_constant = 0;

	ec->topic_predictions.erase();

	float label = ((label_data*) ec->ld)->label;
	((label_data*) ec->ld)->label = FLT_MAX;

	data->base.learn(ec);

	// cout << "L=" << ec->partial_prediction << endl;

	ec->topic_predictions.push_back(ec->partial_prediction);

	for (vector<string>::iterator i = data->pairs.begin();
			i != data->pairs.end(); i++) {
		if (ec->atomics[(int) (*i)[0]].size() > 0
				&& ec->atomics[(int) (*i)[1]].size() > 0) {
			for (size_t k = 1; k <= all->rank; k++) {
				ec->atomics[temp_ind].erase();

				/*
				 cout << "before:";
				 for (unsigned char* x = indices.begin; x != indices.end; x++)
				 for (feature * f = ec->atomics[*x].begin;
				 f != ec->atomics[*x].end; f++) {
				 cout << *x << ":" << f->x << ":" << f->weight_index
				 << ":"
				 << all->reg.weight_vector[f->weight_index & mask]
				 << "  ";
				 }
				 cout << endl;
				 */
				for (feature* f = ec->atomics[(int) (*i)[0]].begin;
						f != ec->atomics[(int) (*i)[0]].end; f++) {
					feature cf;

					cf.weight_index = f->weight_index + k * all->reg.stride;
					cf.x = f->x;

					ec->atomics[temp_ind].push_back(cf);
				}
				// cout << endl;

				ec->indices.erase();
				ec->indices.push_back(temp_ind);

				data->base.learn(ec);
				//cout << "l=" << ec->partial_prediction << endl;
				ec->atomics[temp_ind].erase();
				/*
				 cout << "after:";
				 for (unsigned char* x = indices.begin; x != indices.end; x++)
				 for (feature * f = ec->atomics[*x].begin;
				 f != ec->atomics[*x].end; f++) {
				 cout << *x << ":" << f->x << ":" << f->weight_index
				 << ":"
				 << all->reg.weight_vector[f->weight_index & mask]
				 << "  ";
				 }
				 cout << endl;
				 */
				float x_dot_l = ec->partial_prediction;
				ec->topic_predictions.push_back(ec->partial_prediction);

				for (feature* f = ec->atomics[(int) (*i)[1]].begin;
						f != ec->atomics[(int) (*i)[1]].end; f++) {
					feature cf;

					cf.weight_index = f->weight_index
							+ (all->rank + k) * all->reg.stride;
					cf.x = f->x;
					ec->atomics[temp_ind].push_back(cf);
				}

				ec->indices.erase();
				ec->indices.push_back(temp_ind);

				data->base.learn(ec);

				//cout << "r=" << ec->partial_prediction << endl;

				float x_dot_r = ec->partial_prediction;
				ec->topic_predictions.push_back(ec->partial_prediction);

				quad_constant += (x_dot_l * x_dot_r);
			}
		}
	}

	ec->atomics[temp_ind].erase();
	/*
	 for (unsigned char* i = indices.begin; i != indices.end; i++)
	 for (feature * f = ec->atomics[*i].begin; f != ec->atomics[*i].end;
	 f++) {
	 cout << *i << ":" << f->x << ":" << f->weight_index << ":"
	 << all->reg.weight_vector[f->weight_index & mask] << "  ";
	 }
	 cout << endl;
	 */

	copy_array(ec->indices, indices);
	//cout << "DEBUG tp:       " << ec->topic_predictions[2] << endl;
	((label_data*) ec->ld)->label = label;
	return quad_constant + ec->topic_predictions[0];
}

void debug_weights(v_array<unsigned char> indices, example * ec, vw * all,
		mf * data) {
	size_t mask = all->reg.weight_mask;

	for (unsigned char* i = indices.begin; i != indices.end; i++)
		for (feature * f = ec->atomics[*i].begin; f != ec->atomics[*i].end;
				f++) {
			cout << *i << ":" << f->x << ":" << f->weight_index << ":"
					<< all->reg.weight_vector[f->weight_index & mask] << "  ";
		}

	for (vector<string>::iterator i = data->pairs.begin();
			i != data->pairs.end(); i++) {
		if (ec->atomics[(int) (*i)[0]].size() > 0
				&& ec->atomics[(int) (*i)[1]].size() > 0) {
			for (size_t k = 1; k <= all->rank; k++) {
				for (feature* f = ec->atomics[(int) (*i)[0]].begin;
						f != ec->atomics[(int) (*i)[0]].end; f++) {
					cout << "l" << ":" << f->x << ":"
							<< f->weight_index + k * all->reg.stride << ":"
							<< all->reg.weight_vector[(f->weight_index
									+ k * all->reg.stride) & mask] << "  ";
				}

				for (feature* f = ec->atomics[(int) (*i)[1]].begin;
						f != ec->atomics[(int) (*i)[1]].end; f++) {
					cout << "r" << ":" << f->x << ":"
							<< f->weight_index
									+ (all->rank + k) * all->reg.stride << ":"
							<< all->reg.weight_vector[(f->weight_index
									+ (all->rank + k) * all->reg.stride) & mask]
							<< "  ";
				}
			}
		}
	}
	cout << endl;
}

void learn_linear(mf* data, vw* all, example* &ec, float quad_constant,
		unsigned char temp_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	//cout << "DEBUG tp:       " << ec->topic_predictions[2] << endl;
	//cout << "before:";
	//debug_weights(indices, ec, all, data);

	// learn linear part
	((label_data*) ec->ld)->initial = quad_constant;
	// cout << ec->loss << "  " << ec->final_prediction << endl;
	all->sd->contraction = data->lin_contraction;
	data->base.learn(ec);
	data->lin_contraction = all->sd->contraction;
	// cout << ec->loss << "  " << ec->final_prediction << endl;
	((label_data*) ec->ld)->initial = 0;

	//cout << "DEBUG tp:       " << ec->topic_predictions[2] << endl;
	//cout << "after Linear:" << endl;
	inline_predict(data, all, ec, temp_ind);

	//debug_weights(indices, ec, all, data, mask);
}

void learn_left(mf* data, vw* all, example* &ec, float linear_constant,
		unsigned char left_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	// Learn the left part
	//cout << "before:";
	//debug_weights(ec->indices, ec, all, data);

	//cout << "DEBUG tp:       " << ec->topic_predictions[2] << endl;

	// prediction = inline_predict(data, all, ec, temp_ind);
	// quad_constant = prediction - ec->topic_predictions[0];
	// linear_constant = ec->topic_predictions[0];

	ec->indices.erase();
	ec->indices.push_back(left_ind);

	ec->atomics[left_ind].erase();

	for (vector<string>::iterator i = data->pairs.begin();
			i != data->pairs.end(); i++) {
		if (ec->atomics[(int) (*i)[0]].size() > 0) {
			for (size_t k = 1; k <= all->rank; k++) {
				for (feature* f = ec->atomics[(int) (*i)[0]].begin;
						f != ec->atomics[(int) (*i)[0]].end; f++) {
					feature cf;

					cf.weight_index = f->weight_index + k * all->reg.stride;
					// cf.weight_index = f->weight_index + k;
					//cout << "                   tp:"
					//		<< ec->topic_predictions[2 * k] << endl;
					// cout << " F Index " << cf.weight_index << endl;
					cf.x = f->x * ec->topic_predictions[2 * k];
					//cout << "l: " << cf.weight_index << endl;
					ec->atomics[left_ind].push_back(cf);
				}

			}
		}
	}

	/*
	 assert(ec->atomics[left_ind].size() == all->rank * (ec->atomics[(int)'u'].end - ec->atomics[(int)'u'].begin));
	 for (unsigned char * i = ec->indices.begin; i != ec->indices.end; i++)
	 cout << *i << "  ";
	 cout << endl;
	 assert(ec->indices.size() == 1);
	 */
	((label_data*) ec->ld)->initial = linear_constant;
	all->sd->contraction = data->left_contraction;
	data->base.learn(ec);
	data->left_contraction = all->sd->contraction;
	((label_data*) ec->ld)->initial = 0;

	//cout << "weights after L:";
	//debug_weights(indices, ec, all, data);

	copy_array(ec->indices, indices);
	float prediction = inline_predict(data, all, ec, left_ind);

	//cout << "pred after Left:" << prediction << endl;

}

void learn_right(mf* data, vw* all, example* &ec, float linear_constant,
		unsigned char right_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	// Learn the right part
	ec->indices.erase();
	ec->indices.push_back(right_ind);

	//cout << "before:";
	//debug_weights(indices, ec, all, data, mask);

	ec->atomics[right_ind].erase();
	for (vector<string>::iterator i = data->pairs.begin();
			i != data->pairs.end(); i++) {
		if (ec->atomics[(int) (*i)[1]].size() > 0) {
			for (size_t k = 1; k <= all->rank; k++) {

				for (feature* f = ec->atomics[(int) (*i)[1]].begin;
						f != ec->atomics[(int) (*i)[1]].end; f++) {
					feature cf;

					cf.weight_index = f->weight_index
							+ (all->rank + k) * all->reg.stride;

					//cf.weight_index = f->weight_index + all->rank + k;

					cf.x = f->x * ec->topic_predictions[2 * k - 1];
					ec->atomics[right_ind].push_back(cf);
				}

			}
		}
	}

	((label_data*) ec->ld)->initial = linear_constant;
	all->sd->contraction = data->right_contraction;
	data->base.learn(ec);
	data->right_contraction = all->sd->contraction;
	((label_data*) ec->ld)->initial = 0;

	//float prediction = inline_predict(data, all, ec, right_ind);

	//cout << "after:";
	//debug_weights(indices, ec, all, data, mask);

	copy_array(ec->indices, indices);

}
void learn_with_output(void* d, example* ec, bool shouldOutput) {
	mf* data = (mf*) d;
	vw* all = data->all;

	if (command_example(all, ec)) {
		data->base.learn(ec);
		return;
	}

	cout << all->weights_per_problem << endl;
	assert(all->training);

	cout << all->sd->example_number << endl;
	unsigned char left_ind = 'a';
	while (ec->atomics[left_ind].begin != ec->atomics[left_ind].end)
		left_ind++;

	unsigned char right_ind = left_ind + 1;
	while (ec->atomics[right_ind].begin != ec->atomics[right_ind].end)
		right_ind++;

	unsigned char temp_ind = right_ind + 1;
	while (ec->atomics[temp_ind].begin != ec->atomics[temp_ind].end)
		temp_ind++;

	float prediction = inline_predict(data, all, ec, temp_ind);
	//cout << "DEBUG tp:       " << ec->topic_predictions[2] << endl;

	//cout << "label: " << ((label_data*) ec->ld)->label << endl;

	float quad_constant = prediction - ec->topic_predictions[0];
	float linear_constant = ec->topic_predictions[0];

	//cout << "before learning: " << prediction << " = (" << linear_constant
	//		<< " + " << quad_constant << ")" << endl;

	learn_linear(data, all, ec, quad_constant, temp_ind);

	linear_constant = ec->topic_predictions[0];

	learn_left(data, all, ec, linear_constant, left_ind);

	linear_constant = ec->topic_predictions[0];

	learn_right(data, all, ec, linear_constant, right_ind);

	ec->final_prediction = inline_predict(data, all, ec, temp_ind);
	//cout << "after learning: " << ec->final_prediction << endl;

	ec->loss = all->loss->getLoss(all->sd, ec->final_prediction,
			((label_data*) ec->ld)->label) * ((label_data*) ec->ld)->weight;

}

void learn(void* d, example* ec) {
	learn_with_output(d, ec, false);
}

void finish(void* data) {
	mf* o = (mf*) data;
	o->base.finish();
	free(o);
}

void drive(vw* all, void* d) {
	example* ec = NULL;

	while (true) {
		if ((ec = VW::get_example(all->p)) != NULL) //blocking operation.
				{
			learn(d, ec);
			return_simple_example(*all, ec);
		} else if (parser_done(all->p))
			return;
		else
			; //busywait when we have predicted on all examples but not yet trained on all.
	}
}

learner setup(vw& all) {
	mf* data = (mf*) calloc(1, sizeof(mf));

	data->base = all.l;
	data->all = &all;

	data->lin_contraction = all.sd->contraction;
	data->left_contraction = all.sd->contraction;
	data->right_contraction = all.sd->contraction;

	data->pairs = all.pairs;

	all.pairs.clear();

	learner l = { data, drive, learn, finish, all.l.sl };

	return l;
}
}
