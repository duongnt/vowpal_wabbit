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

float inline_predict(mf* data, vw* all, example* &ec, unsigned char temp_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	float quad_constant = 0;

	ec->topic_predictions.erase();

	float label = ((label_data*) ec->ld)->label;
	((label_data*) ec->ld)->label = FLT_MAX;

	data->base.learn(ec);

	ec->topic_predictions.push_back(ec->partial_prediction);

	for (vector<string>::iterator i = data->pairs.begin();
			i != data->pairs.end(); i++) {
		if (ec->atomics[(int) (*i)[0]].size() > 0
				&& ec->atomics[(int) (*i)[1]].size() > 0) {
			for (size_t k = 1; k <= all->rank; k++) {
				ec->atomics[temp_ind].erase();

				for (feature* f = ec->atomics[(int) (*i)[0]].begin;
						f != ec->atomics[(int) (*i)[0]].end; f++) {
					feature cf;

					cf.weight_index = f->weight_index + k * all->reg.stride;
					cf.x = f->x;

					ec->atomics[temp_ind].push_back(cf);
				}

				ec->indices.erase();
				ec->indices.push_back(temp_ind);

				data->base.learn(ec);

				ec->atomics[temp_ind].erase();

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

				float x_dot_r = ec->partial_prediction;
				ec->topic_predictions.push_back(ec->partial_prediction);

				quad_constant += (x_dot_l * x_dot_r);
			}
		}
	}

	ec->atomics[temp_ind].erase();

	copy_array(ec->indices, indices);

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

	((label_data*) ec->ld)->initial = quad_constant;

	all->sd->contraction = data->lin_contraction;
	data->base.learn(ec);
	data->lin_contraction = all->sd->contraction;

	((label_data*) ec->ld)->initial = 0;

	inline_predict(data, all, ec, temp_ind);
}

void learn_left(mf* data, vw* all, example* &ec, float linear_constant,
		unsigned char left_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

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
					cf.x = f->x * ec->topic_predictions[2 * k];

					ec->atomics[left_ind].push_back(cf);
				}
			}
		}
	}

	((label_data*) ec->ld)->initial = linear_constant;
	all->sd->contraction = data->left_contraction;
	data->base.learn(ec);
	data->left_contraction = all->sd->contraction;
	((label_data*) ec->ld)->initial = 0;

	copy_array(ec->indices, indices);
	float prediction = inline_predict(data, all, ec, left_ind);
}

void learn_right(mf* data, vw* all, example* &ec, float linear_constant,
		unsigned char right_ind) {
	v_array<unsigned char> indices;
	copy_array(indices, ec->indices);

	ec->indices.erase();
	ec->indices.push_back(right_ind);

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

	copy_array(ec->indices, indices);

}
void learn_with_output(void* d, example* ec, bool shouldOutput) {
	mf* data = (mf*) d;
	vw* all = data->all;

	if (command_example(all, ec)) {
		data->base.learn(ec);
		return;
	}

	assert(all->training);

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

	float quad_constant = prediction - ec->topic_predictions[0];

	learn_linear(data, all, ec, quad_constant, temp_ind);

	float linear_constant = ec->topic_predictions[0];

	learn_left(data, all, ec, linear_constant, left_ind);

	learn_right(data, all, ec, linear_constant, right_ind);

	ec->final_prediction = inline_predict(data, all, ec, temp_ind);

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
