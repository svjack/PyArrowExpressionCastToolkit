#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:19:23 2021

@author: svjack
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.csv as pc
from pyarrow import hdfs
from pyarrow.hdfs import HadoopFileSystem
from pyarrow.filesystem import LocalFileSystem
import pyarrow.lib as lib
from pyarrow.lib import UnionArray, Array, ChunkedArray
from pyarrow.dataset import _union_dataset
from pyarrow.parquet import ParquetFile
from pyarrow.parquet import ParquetDataset
from pyarrow.parquet import _ParquetDatasetV2
from pyarrow.dataset import dataset
from pyarrow.dataset import Dataset
from pyarrow.dataset import Expression
from pyarrow.dataset import _get_partition_keys
from pyarrow.dataset import HivePartitioning
from pyarrow.lib import DictionaryType, DataType

import pandas as pd
import re
from functools import reduce
from functools import partial
import os
from glob import glob
import numpy as np
from numpy import vectorize
import json
import ast

from pyarrow.lib import *
_type_aliases = {
    'null': null,
    'bool': bool_,
    'boolean': bool_,
    'i1': int8,
    'int8': int8,
    'i2': int16,
    'int16': int16,
    'i4': int32,
    'int32': int32,
    'i8': int64,
    'int64': int64,
    'u1': uint8,
    'uint8': uint8,
    'u2': uint16,
    'uint16': uint16,
    'u4': uint32,
    'uint32': uint32,
    'u8': uint64,
    'uint64': uint64,
    'f2': float16,
    'halffloat': float16,
    'float16': float16,
    'f4': float32,
    'float': float32,
    'float32': float32,
    'f8': float64,
    'double': float64,
    'float64': float64,
    'string': string,
    'str': string,
    'utf8': string,
    'binary': binary,
    'large_string': large_string,
    'large_str': large_string,
    'large_utf8': large_string,
    'large_binary': large_binary,
    'date32': date32,
    'date64': date64,
    'date32[day]': date32,
    'date64[ms]': date64,
    'time32[s]': time32('s'),
    'time32[ms]': time32('ms'),
    'time64[us]': time64('us'),
    'time64[ns]': time64('ns'),
    'timestamp[s]': timestamp('s'),
    'timestamp[ms]': timestamp('ms'),
    'timestamp[us]': timestamp('us'),
    'timestamp[ns]': timestamp('ns'),
    'duration[s]': duration('s'),
    'duration[ms]': duration('ms'),
    'duration[us]': duration('us'),
    'duration[ns]': duration('ns'),
}


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 255)

import operator

def _check_contains_null(val):
    if isinstance(val, bytes):
        for byte in val:
            if isinstance(byte, bytes):
                compare_to = chr(0)
            else:
                compare_to = 0
            if byte == compare_to:
                return True
    elif isinstance(val, str):
        return '\x00' in val
    return False


def _check_filters(filters, check_null_strings=True):
    """
    Check if filters are well-formed.
    """
    if filters is not None:
        if len(filters) == 0 or any(len(f) == 0 for f in filters):
            raise ValueError("Malformed filters")
        if isinstance(filters[0][0], str):
            # We have encountered the situation where we have one nesting level
            # too few:
            #   We have [(,,), ..] instead of [[(,,), ..]]
            filters = [filters]
        if check_null_strings:
            for conjunction in filters:
                for col, op, val in conjunction:
                    if (
                        isinstance(val, list) and
                        all(_check_contains_null(v) for v in val) or
                        _check_contains_null(val)
                    ):
                        raise NotImplementedError(
                            "Null-terminated binary strings are not supported "
                            "as filter values."
                        )
    return filters

def _filters_to_expression(filters):
    """
    Check if filters are well-formed.

    See _DNF_filter_doc above for more details.
    """
    import pyarrow.dataset as ds
    if isinstance(filters, ds.Expression):
        return filters
    filters = _check_filters(filters, check_null_strings=False)
    def convert_single_predicate(col, op, val):
        field = ds.field(col)
        if op == "=" or op == "==":
            return field == val
        elif op == "!=":
            return field != val
        elif op == '<':
            return field < val
        elif op == '>':
            return field > val
        elif op == '<=':
            return field <= val
        elif op == '>=':
            return field >= val
        elif op == 'in':
            return field.isin(val)
        elif op == 'not in':
            return ~field.isin(val)
        else:
            raise ValueError(
                '"{0}" is not a valid operator in predicates.'.format(
                    (col, op, val)))
    disjunction_members = []
    for conjunction in filters:
        conjunction_members = [
            convert_single_predicate(col, op, val)
            for col, op, val in conjunction
        ]
        disjunction_members.append(reduce(operator.and_, conjunction_members))
    return reduce(operator.or_, disjunction_members)

def gen_filter_type_cast_df(filters_objects, filters_types):
    #print(list(map(id, [filters_objects, filters_types])))
    assert len(filters_objects) == len(filters_types)
    def flatten_func(collections):
        if list(set(map(type ,collections)))[0] == type([]):
            return reduce(lambda a, b: a + b, collections)
        return collections
    filters_objects_flatten, filters_types_flatten = map(flatten_func, [filters_objects, filters_types])
    assert len(filters_objects_flatten) == len(filters_types_flatten)
    assert set(map(len, filters_objects_flatten)) == set([3])
    assert set(map(len, filters_types_flatten)) == set([2])
    assert set(map(lambda t: t[0], filters_objects_flatten)) == set(map(lambda t: t[0], filters_types_flatten)) 
    assert len(set(map(lambda t: t[0], filters_types_flatten))) == len(list(map(lambda t: t[0], filters_types_flatten)))
    return pd.DataFrame(filters_types_flatten, columns = ["key", "cast_type"])

def _filters_to_expression_cast_type(filters, filters_types):
    """
    Check if filters are well-formed.

    See _DNF_filter_doc above for more details.
    """
    #### this func use difference key name for partitions
    ## if partition have nest construction, with identi name, can not pass.
    #type_cast_lookup_df = gen_filter_type_cast_df(filters_objects, filters_types)
    type_cast_lookup_df = gen_filter_type_cast_df(filters, filters_types)
    import pyarrow.dataset as ds
    if isinstance(filters, ds.Expression):
        return filters
    filters = _check_filters(filters, check_null_strings=False)
    def convert_single_predicate(col, op, val):
        field = ds.field(col)
        cast_type = None
        if col in type_cast_lookup_df["key"].tolist():
            cast_type = type_cast_lookup_df[type_cast_lookup_df["key"] == col]["cast_type"].iloc[0]
            assert type(cast_type) == type("")
        if cast_type is not None:
            field = field.cast(cast_type)
        if op == "=" or op == "==":
            return field == val
        elif op == "!=":
            return field != val
        elif op == '<':
            return field < val
        elif op == '>':
            return field > val
        elif op == '<=':
            return field <= val
        elif op == '>=':
            return field >= val
        elif op == 'in':
            return field.isin(val)
        elif op == 'not in':
            return ~field.isin(val)
        else:
            raise ValueError(
                '"{0}" is not a valid operator in predicates.'.format(
                    (col, op, val)))
    disjunction_members = []
    for conjunction in filters:
        conjunction_members = [
            convert_single_predicate(col, op, val)
            for col, op, val in conjunction
        ]
        disjunction_members.append(reduce(operator.and_, conjunction_members))
    return reduce(operator.or_, disjunction_members)


def retrieve_inner_indices(filters_objects):
    def flatten_func(collections):
        if list(set(map(type ,collections)))[0] == type([]):
            return reduce(lambda a, b: a + b, collections)
        return collections
    filters_objects_flattened = flatten_func(filters_objects)
    filters_objects_flattened_str = list(map(str ,filters_objects_flattened))
    req = str(filters_objects)
    for idx, filters_objects_str in enumerate(filters_objects_flattened_str):
        req = req.replace(filters_objects_str, str(idx))
    assert max(np.asarray(eval(req)).reshape([-1])) + 1 == len(filters_objects_flattened)
    return eval(req), filters_objects_flattened

def reproduce_min_part(expression_format, expression_collections):
    assert max(np.asarray(expression_format).reshape([-1])) + 1 == len(expression_collections)
    shape = np.asarray(expression_format).shape
    need_shape = tuple(list(shape[:-1]) + [1])
    expression_collections_str = list(map(str ,expression_collections))
    def eval_with_import(str_):
        #### this field import some type cast such as Timestamp
        Timestamp = pd.Timestamp
        return eval(str_)
    vectorize_eval = vectorize(eval_with_import)
    reproduce_tuple = tuple(map(lambda ele: vectorize_eval(np.asarray([ele]).reshape(need_shape)), expression_collections_str))
    shape_set = set(map(lambda x: x.shape, reduce(lambda a, b: a + b ,reproduce_tuple)))
    assert len(shape_set) == 1
    shape = list(shape_set)[0]
    def retrieve_element(input_, shape):
        def retrieve_first(input_):
            return input_[0]
        for i in range(len(shape)):
            input_ = retrieve_first(input_)
        return input_
    final_c = list(map(lambda tuple_: eval("[" * len(shape) + str(tuple(map(partial(retrieve_element, shape = shape), tuple_))) + "]" * len(shape)) , reproduce_tuple))
    return final_c

def reproduce_min_expression_by_main_input(filters_objects, filters_types, return_dict_format = False):
    min_expression_format, min_expression_collections = retrieve_inner_indices(filters_objects)
    reproduce_tuple_objects = reproduce_min_part(min_expression_format, min_expression_collections)
    min_expression_format, min_expression_collections = retrieve_inner_indices(filters_types)
    reproduce_tuple_types = reproduce_min_part(min_expression_format, min_expression_collections)
    assert len(reproduce_tuple_objects) == len(reproduce_tuple_types)
    req_list = list(zip(*[reproduce_tuple_objects, reproduce_tuple_types]))
    if not return_dict_format:
        return req_list
    def get_first(input_):
        if type(input_) != type(""):
            return get_first(input_[0])
        else:
            return input_
    keys = list(map(get_first, req_list))
    req_dict = {}
    for i in range(len(req_list)):
        k = keys[i]
        ele = req_list[i]
        req_dict[k] = ele
    return req_dict
    
def components_maintain_filters_to_expression(filters_objects, filters_types):
    #### in the construction of total main expression
    ## also save sub_part as min_expression in dictionary
    ## by shape decomposition by numpy, and metain same constrution with main_expression
    ## by vecterize 
    ## if not use cast, the code can be more simple.
    assert len(filters_objects) == len(filters_types)
    main_expression = _filters_to_expression_cast_type(filters_objects, filters_types)
    min_expression_dict = reproduce_min_expression_by_main_input(filters_objects, filters_types, return_dict_format = True)
    ##### min_expression all have objs and types as ele
    assert set(map(len ,min_expression_dict.values())) == set([len([filters_objects, filters_types])])
    #return min_expression_dict
    min_expression_dict = dict(map(lambda t2: (t2[0], _filters_to_expression_cast_type(*t2[1])), min_expression_dict.items()))
    assert all(map(lambda min_exp: min_exp.assume(main_expression).__str__() == "true:bool" ,min_expression_dict.values()))
    return main_expression, min_expression_dict

def produce_key_with_cast_expression(expression_contain_cast):
    assert isinstance(expression_contain_cast, Expression)
    assert "cast" in expression_contain_cast.__str__()
    # ((cast job to string) == migsql_hive_main:string)
    cast_pattern = r"\(\(cast (.*) to (.*)\)(.*)\)"
    key, cast_type, _ = re.findall(cast_pattern, expression_contain_cast.__str__())[0]
    return key, cast_type, _

def retrieve_partition_val_by_key(p_path ,keys = []):
    #### limit keys range may save some m or s
    p_dataset_parquet = _ParquetDatasetV2(p_path)
    pieces = p_dataset_parquet.pieces
    def merge_a_b_dict(a_dict, b_dict):
        assert set(a_dict.keys()) == set(b_dict.keys())
        req = {}
        for k in a_dict.keys():
            req[k] = a_dict[k].union(b_dict[k])
        return req
    if not keys:
        return reduce(merge_a_b_dict, map(lambda p: dict(map(lambda t2: (t2[0], set([t2[1]])),_get_partition_keys(p.partition_expression).items())), pieces))
    else:
        assert type(keys) == type([])
        return reduce(merge_a_b_dict, map(lambda p: dict(map(lambda t2: (t2[0], set([t2[1]])), filter(lambda tt2: tt2[0] in keys ,_get_partition_keys(p.partition_expression).items()))), pieces))


#### set cast_type_type_dict with _type_aliases 
## will map perform type cast in below order:
## string -> python -> pyarrow -> pandas
## all type map to pandas type space to compare and filter
## and maintain mapping string -> pandas (kp2kpcast) \ pandas -> string (kpcast2kp)
## to final perform filter.
## this will make _ParquetDatasetV2PartitionCast prevemt param setting of cast_type_func_dict
## if partition string are all will formated,
## so, if in my example not well formatted, backup_time
## may use some sense of  cast_type_ast_wrapper_dict
## {"timestamp[ms]": lambda x: x[:x.find(">")]}
## so can only use wrapper dict to pre-process some not-well-formatted
### so the feature of this class is abviously Cast and Well-Formatted
### TODO: _ParquetDatasetV2PartitionCast -> _ParquetDatasetV2PartitionCastWellFormatted

#### can only use cast_type_ast_wrapper_dict as a format editor.
#### doesn't need cast config dict in cls level.
def produce_partition_mapper_with_cast_expression_by_pa(p_path ,expression_contain_cast, cast_type_type_dict = {}, cast_type_ast_wrapper_dict = {}):
    #print(produce_key_with_cast_expression(expression_contain_cast))
    key, cast_type, _ = produce_key_with_cast_expression(expression_contain_cast)
    assert cast_type in cast_type_type_dict
    cast_type_type = cast_type_type_dict[cast_type]
    if callable(cast_type_type):
        #### cast_type_type may be cls or func, if func, init it to type cls
        cast_type_type = cast_type_type()
    #### default wrapper identity
    ast_wrapper = lambda x: x
    if cast_type_type in cast_type_ast_wrapper_dict:
        ast_wrapper = cast_type_ast_wrapper_dict[cast_type_type]
    #### all cast to pandas, so the filter conpare is equalent with func not by_pa.
    def cast_by_literal_eval(val):
        try:
            return ast.literal_eval(val)
        except:
            return val
    def cast_type_func(x, use_ast = True):
        assert type(x) == type([])
        if use_ast:
            #### with cast_by_literal_eval guarantte elements in x cast from string to proper 
            ### python type firstly to use in pa.array cast
            x = list(map(lambda y: cast_by_literal_eval(ast_wrapper(y)), x))
        assert type(x) == type([])
        return pa.array(x).cast(cast_type_type).to_pandas().tolist()
    key_partitions_set = retrieve_partition_val_by_key(p_path, keys = [key])[key]
    key_partitions_list = list(key_partitions_set)
    kp2kpcast = dict(zip(*[key_partitions_list ,cast_type_func(key_partitions_list)]))
    #### use kpcast as key, must can hash
    assert hash(list(kp2kpcast.values())[0])
    kpcast2kp = dict(map(lambda t2: (t2[1], t2[0]), kp2kpcast.items()))
    assert len(kp2kpcast) == len(kpcast2kp)
    return key ,kp2kpcast, kpcast2kp, _

def condition_on_kpcast2kp_by_pa(kpcast2kp, condition, cast_type_type_dict = {}, cast_type_ast_wrapper_dict = {}):
    if ":" in condition:
        condition_p = condition[:condition.find(":")]
        cast_type = condition[condition.find(":") + 1:]
    condition_parsed = list(filter(lambda y: y ,map(lambda x: x.strip() ,condition_p.split(" "))))
    assert len(condition_parsed) == 2
    assert set(map(type, condition_parsed)) == set([type("")])
    op, val = condition_parsed
    assert cast_type in cast_type_type_dict
    cast_type_type = cast_type_type_dict[cast_type]
    if callable(cast_type_type):
        cast_type_type = cast_type_type()
    ast_wrapper = lambda x: x
    if cast_type_type in cast_type_ast_wrapper_dict:
        ast_wrapper = cast_type_ast_wrapper_dict[cast_type_type]
    #### all cast to pandas, so the filter conpare is equalent with func not by_pa.
    def cast_by_literal_eval(val):
        try:
            return ast.literal_eval(val)
        except:
            return val
    def cast_type_func(x, use_ast = True):
        assert type(x) == type([])
        if use_ast:
            #### with cast_by_literal_eval guarantte elements in x cast from string to proper 
            ### python type firstly to use in pa.array cast
            x = list(map(lambda y: cast_by_literal_eval(ast_wrapper(y)), x))
        assert type(x) == type([])
        return pa.array(x).cast(cast_type_type).to_pandas().tolist()  
    val = cast_type_func([val])[0]
    def convert_single_predicate(col, op, val):
        field = col
        if op == "=" or op == "==":
            return field == val
        elif op == "!=":
            return field != val
        elif op == '<':
            return field < val
        elif op == '>':
            return field > val
        elif op == '<=':
            return field <= val
        elif op == '>=':
            return field >= val
        elif op == 'in':
            return field.isin(val)
        elif op == 'not in':
            return ~field.isin(val)
        else:
            raise ValueError(
                '"{0}" is not a valid operator in predicates.'.format(
                    (col, op, val)))
    kpcast2kp_filtered_by_op_val = dict(filter(lambda t2: convert_single_predicate(t2[0], op, val), kpcast2kp.items()))
    return kpcast2kp_filtered_by_op_val

def map_kp_filtered_to_union_expression(key ,kpcast2kp_filtered_by_op_val, by = "piece", p_path = None, 
                                        min_expression_format = [[0]]):
    #### if use get_fragments in this method may have dataset filtered directly, 
    ## but this only retrieve total expression
    assert by in ["piece", "partition", "directly"]
    
    if by == "piece":
        kp_filtered_lookup_list = list(map(lambda kp: "{}={}".format(key, kp) ,kpcast2kp_filtered_by_op_val.values()))
        assert p_path is not None
        assert type(p_path) == type("") and os.path.exists(p_path)
        p_dataset_parquet = _ParquetDatasetV2(p_path)
        pieces = p_dataset_parquet.pieces
        return reduce(lambda a, b: a.__or__(b) ,map(lambda pp: pp.partition_expression ,filter(lambda p: sum(map(lambda kp: kp in p.path, kp_filtered_lookup_list)), pieces)))
    elif by == "partition":
        assert p_path is not None
        assert type(p_path) == type("") and os.path.exists(p_path)
        p_dataset_parquet = _ParquetDatasetV2(p_path)
        schema = p_dataset_parquet.schema
        def retrieve_value_type(type_):
            assert sum(map(lambda y: isinstance(type_, y), [DictionaryType, DataType]))
            ### above have inher relation
            if hasattr(type_, "value_type"):
                return type_.value_type
            else:
                return type_
        name_type_dict = dict(map(lambda n: (n, retrieve_value_type(schema.types[schema.get_field_index(n)])), schema.names))
        partitioning = HivePartitioning(pa.schema(list(filter(lambda t2: t2[0] == key ,name_type_dict.items()))))
        return reduce(lambda a, b: a.__or__(b) ,map(lambda kp: partitioning.parse("{}={}".format(key, kp)) ,kpcast2kp_filtered_by_op_val.values()))
    else:
        #### only one key, min_expression_format == [[0]]
        assert min_expression_format == [[0]]
        op = "in"
        min_expression_collections = [(key, op, list(kpcast2kp_filtered_by_op_val.values()))]
        def reproduce_min_part_with_in_op(min_expression_format, min_expression_collections):
            assert len(min_expression_collections) == 1
            assert type(min_expression_collections[0]) == type((1,))
            shape = np.asarray(min_expression_format).shape
            return eval("[" * len(shape) + str(min_expression_collections[0]) + "]" * len(shape))
        reproduce_tuple_objects = reproduce_min_part_with_in_op(min_expression_format, min_expression_collections)
        return _filters_to_expression(reproduce_tuple_objects)

def collection_unfold_checker(filters_objects):
    min_expression_format, min_expression_collections = retrieve_inner_indices(filters_objects)
    def elements_checker(min_expression_element):
        assert type(min_expression_element) == type((1,))
        assert len(min_expression_element) == 3
        if type(min_expression_element[-1]) == type([]):
            col, op, val = min_expression_element
            assert op in ["in", "not in"]
            element_op = "=" if op == "in" else "!="
            req = list(map(lambda val: (col, element_op, val), min_expression_element[-1]))
        else:
            req = min_expression_element
        if type(min_expression_element[-1]) == type([]):
            assert type(req) == type([])
        else:
            assert type(req) == type((1,))
        return req
    nest_min_expression_collection = []
    for min_expression_element in min_expression_collections:
        nest_min_expression_collection.append(elements_checker(min_expression_element))
    assert len(nest_min_expression_collection) == len(min_expression_collections)
    ##### element with list will cartesian product with others
    nest_min_expression_collection_df = pd.DataFrame(list(map(str, nest_min_expression_collection))).T
    def eval_with_import(str_):
        #### this field import some type cast such as Timestamp
        Timestamp = pd.Timestamp
        return eval(str_)
    type_mapping_list = list(map(type, nest_min_expression_collection))
    def retrieve_merge_op(ele):
        assert type(ele) in [type([]), type((1,))]
        op = None
        if type(ele) == type([]):
            assert len(set(map(lambda e: e[1], ele))) == 1
            element_op = list(set(map(lambda e: e[1], ele)))[0]
            op = "or" if element_op == "=" else "and"
        return op
    merge_op_list = list(map(retrieve_merge_op, nest_min_expression_collection))
    merge_op_dict = dict(map(lambda ele: (ele[0] if type(ele) == type((1,)) else ele[0][0] ,retrieve_merge_op(ele)), nest_min_expression_collection))
    assert len(merge_op_list) == len(merge_op_dict)
    #### cartesian product by explode
    nest_min_expression_collection_df = nest_min_expression_collection_df.applymap(eval_with_import)
    for idx ,type_mapping in enumerate(type_mapping_list):
        assert type_mapping in [type([]), type((1,))]
        if type_mapping == type([]):
            nest_min_expression_collection_df = nest_min_expression_collection_df.explode(idx)
    nest_min_expression_collection_df.values.tolist()
    #### a series every element is a dict , the merge op above them 
    #### defined by merge_op_list
    unfold_list = list(map(
            lambda element_level_min_expression_collections: 
        {"min_expression_collections":  element_level_min_expression_collections,
            "min_expression_format": min_expression_format
            }
            , nest_min_expression_collection_df.values.tolist()))
    return unfold_list, merge_op_dict
    
def reproduce_min_expression_by_main_input_unfold(filters_objects, filters_types, return_dict_format = False):
    min_expression_format, min_expression_collections = retrieve_inner_indices(filters_types)
    reproduce_tuple_types = reproduce_min_part(min_expression_format, min_expression_collections)
    def reproduce_tuple_objects_AND_reproduce_tuple_types_TO_req_dict(reproduce_tuple_objects, reproduce_tuple_types):
        assert len(reproduce_tuple_objects) == len(reproduce_tuple_types)
        req_list = list(zip(*[reproduce_tuple_objects, reproduce_tuple_types]))
        if not return_dict_format:
            return req_list
        def get_first(input_):
            if type(input_) != type(""):
                return get_first(input_[0])
            else:
                return input_
        keys = list(map(get_first, req_list))
        req_dict = {}
        for i in range(len(req_list)):
            k = keys[i]
            ele = req_list[i]
            req_dict[k] = ele
        return req_dict
    unfold_list, merge_op_dict = collection_unfold_checker(filters_objects)
    merge_op_list = list(merge_op_dict.values())
    if len(unfold_list) == 1:
        #### ori condition not need merge
        assert set(merge_op_list) == set([None])
    else:
        #### need merge with merge_op_list
        pass
    reproduce_tuple_objects_unfold_list = list(map(lambda dict_: reproduce_min_part(dict_["min_expression_format"], dict_["min_expression_collections"]), unfold_list))
    req = []
    for reproduce_tuple_objects in reproduce_tuple_objects_unfold_list:
        req.append(reproduce_tuple_objects_AND_reproduce_tuple_types_TO_req_dict(reproduce_tuple_objects, reproduce_tuple_types))
    return {"req": req, "merge_op_dict": merge_op_dict}
    

def components_maintain_filters_to_expression_unfold(filters_objects, filters_types):
    #### in the construction of total main expression
    ## also save sub_part as min_expression in dictionary
    ## by shape decomposition by numpy, and metain same constrution with main_expression
    ## by vecterize 
    ## if not use cast, the code can be more simple.
    assert len(filters_objects) == len(filters_types)
    main_expression = _filters_to_expression_cast_type(filters_objects, filters_types)
    min_expression_dict = reproduce_min_expression_by_main_input_unfold(filters_objects, filters_types, return_dict_format=True)
    assert set(min_expression_dict.keys()) == set(["req", "merge_op_dict"])
    assert reduce(lambda a, b: a.union(b) ,map(lambda x: set(map(len ,x.values())) ,min_expression_dict["req"])) == set([len([filters_objects, filters_types])])
    req = min_expression_dict["req"]
    merge_op_dict = min_expression_dict["merge_op_dict"]
    def produce_single_min_expression_dict(req_ele):
        min_expression_dict_single = dict(map(lambda t2: (t2[0], _filters_to_expression_cast_type(*t2[1])), req_ele.items()))
        #### because expression dnot have logic simplify , so assume assert can't pass.
        #assert all(map(lambda min_exp: min_exp.assume(main_expression).__str__() == "true:bool" ,min_expression_dict_single.values()))
        return min_expression_dict_single
    list_of_min_expression_dict = list(map(produce_single_min_expression_dict, req))
    return main_expression, list_of_min_expression_dict, merge_op_dict

#### above three func use map reduce manner to support "in" and "not in"
#### compare, 
#### transform "in" into __or__(explode(col) = ), "not in" into __and__(explode(col) != )
#### in cartesian product manner and collect these expression in different collection
#### final map reduce (by different op) in blow class
#### this seems support all pa cast and all cmp op in the module.
class _ParquetDatasetV2PartitionCastWellFormatted(_ParquetDatasetV2):
    #### declare eval alias
    Timestamp = pd.Timestamp
    def __init__(self, path_or_paths,
                 filters_objects, filters_types, 
                 filesystem=None, filters=None,
                 partitioning="hive", read_dictionary=None, buffer_size=None,
                 memory_map=False, ignore_prefixes=None, 
                 cast_type_ast_wrapper_dict = {
        "timestamp[ms]": lambda x: "%d" % (pd.to_datetime([x]).astype(int) / (10 ** 6))[0]
        },
            expression_construct_by = "directly",
            expression_merge_op = "and",
                  **kwargs):
        assert type(path_or_paths) in [type(""), type([])]
        if type(path_or_paths) == type([]):
            return list(map(lambda p: _ParquetDatasetV2PartitionCastWellFormatted(p, 
                                        filters_objects, filters_types,
                                        filesystem, filters,
                 partitioning, read_dictionary, 
                 buffer_size,
                 memory_map, ignore_prefixes,
                 cast_type_ast_wrapper_dict,
                 expression_construct_by,
                 expression_merge_op,
                 **kwargs
                 ), path_or_paths))
        assert os.path.exists(p_path)
        assert type(filters_objects) == type([]) and type(filters_types) == type([])
        assert expression_construct_by in ["piece", "partition", "directly"]
        assert expression_merge_op in ["and", "or"]
        #### input params
        self.p_path = path_or_paths
        self.filters_objects = filters_objects
        self.filters_types = filters_types
        self.expression_construct_by = expression_construct_by
        self.expression_merge_op = expression_merge_op
        #### consider following cast expression :
        # col op val:format
        # ((cast backup_time to timestamp[ms]) > 1610187854000000:timestamp[us])
        ### perform string -> python -> pyarrow -> pandas space mapping
        ### cast_type_ast_wrapper_dict wrap ast func between string to python (string -> ast_wrapper -> ast_eval -> python)
        self.cast_type_ast_wrapper_dict = cast_type_ast_wrapper_dict
        #### params generate
        self.main_expression = None
        self.min_expression_dict = None
        self.min_kp_expression_list = []
        self.min_kp_expression_and_list = []
        self.min_kp_expression_or_list = []
        self.expression_merge_op_func = lambda a, b: a.__and__(b) if self.expression_merge_op == "and" else lambda a, b: a.__or__(b)
        self.expression_merge_op_func_dict = {
                "and": lambda a, b: a.__and__(b),
                "or": lambda a, b: a.__or__(b)
                }
        self.final_expression = None
        self.dataset_ = None
        #### func running range
        self.init()
        self.map()
        self.reduce()
        #### base construction
        if filters is not None:
            assert type(filters) == type([]) or isinstance(filters, Expression)
            if type(filters) == type([]):
                filters = _filters_to_expression(filters)
        assert filters is None or isinstance(filters, Expression)
        if isinstance(filters, Expression):
            filters = filters and self.final_expression
        else:
            filters = self.final_expression
        assert isinstance(filters, Expression)
        super(_ParquetDatasetV2PartitionCastWellFormatted, self).__init__(path_or_paths, 
             filesystem, filters,
                 partitioning, read_dictionary, 
                 buffer_size,
                 memory_map, ignore_prefixes, **kwargs)
    def decomposition_expression_collection(self, verbose = True):
        #self.main_expression, self.min_expression_dict = components_maintain_filters_to_expression(self.filters_objects, self.filters_types)
        self.main_expression, self.list_of_min_expression_dict, self.merge_op_dict = components_maintain_filters_to_expression_unfold(self.filters_objects, self.filters_types)
        assert type(verbose) in [type(0), type(True)]
        verbose_int = int(verbose)
        if verbose_int:
            print("main_expression :")
            print(self.main_expression.__str__())
            for min_expression_dict in self.list_of_min_expression_dict:
                if min_expression_dict:
                    print("has min_expression :")
                    for name, exp in min_expression_dict.items():
                        print("expression name {}".format(name))
                        print(exp.__str__())
                print("-" * 100)
    def single_min_expression_producer(self ,min_expression, verbose = True):
        key ,kp2kpcast, kpcast2kp, _ = produce_partition_mapper_with_cast_expression_by_pa(self.p_path ,min_expression, 
                                                            cast_type_type_dict = _type_aliases, 
                                                            cast_type_ast_wrapper_dict = self.cast_type_ast_wrapper_dict)
        kpcast2kp_filtered_by_op_val = condition_on_kpcast2kp_by_pa(kpcast2kp, _, 
                                     cast_type_type_dict = _type_aliases, 
                                     cast_type_ast_wrapper_dict = {})
        exp = map_kp_filtered_to_union_expression(key, kpcast2kp_filtered_by_op_val, by = self.expression_construct_by, p_path = self.p_path)
        verbose_int = int(verbose)
        if verbose_int:
            print("single exp: {}".format(exp))
        return exp
    def multi_expression_merger(self ,min_kp_expression_list, expression_merge_op_func):
        return reduce(expression_merge_op_func, min_kp_expression_list)
    def init(self):
        self.decomposition_expression_collection()
    def map(self):
        assert self.merge_op_dict is not None
        for min_expression_dict in self.list_of_min_expression_dict:
            for name, exp in min_expression_dict.items():
                assert name in self.merge_op_dict
                if self.merge_op_dict[name] is None:
                    self.min_kp_expression_list.append(self.single_min_expression_producer(exp))
                else:
                    assert self.merge_op_dict[name] in ["and", "or"]
                    if self.merge_op_dict[name] == "and":
                        self.min_kp_expression_and_list.append(self.single_min_expression_producer(exp))
                    else:
                        self.min_kp_expression_or_list.append(self.single_min_expression_producer(exp))
    def reduce(self, verbose = True):
        #self.final_expression = self.multi_expression_merger(self.min_kp_expression_list)
        req = []
        if self.min_kp_expression_list:
            final_expression_default = self.multi_expression_merger(self.min_kp_expression_list, self.expression_merge_op_func)
            req.append(final_expression_default)
        if self.min_kp_expression_and_list:
            final_expression_and = self.multi_expression_merger(self.min_kp_expression_and_list, self.expression_merge_op_func_dict["and"])
            req.append(final_expression_and)
        if self.min_kp_expression_or_list:
            final_expression_or = self.multi_expression_merger(self.min_kp_expression_or_list, self.expression_merge_op_func_dict["or"])
            req.append(final_expression_or)
        self.final_expression = self.multi_expression_merger(req, self.expression_merge_op_func)
        verbose_int = int(verbose)
        if verbose_int:
            print("final exp: {}".format(self.final_expression))

'''
### ((cast event to string) == transaction:string)
### ((cast event to string) == transaction:string)
filters = [[("(cast event to string)", "=", "transaction"), ]]
_filters_to_expression(filters)
ds = _ParquetDatasetV2(
        p_path,
        filters = _filters_to_expression(filters)
        )

ds.read().to_pandas()
'''

#### download event.csv from
### https://www.kaggle.com/retailrocket/ecommerce-dataset?select=events.csv
events_df = pd.read_csv("/home/svjack/temp_dir/events.csv")

events_df["time"] = pd.to_datetime(events_df["timestamp"]).astype(str)
events_df["time_two_pos"] = events_df["time"].map(lambda x: str(x)[:str(x).find(".") + 2] + "0" * (28 - len(str(x)[:str(x).find(".") + 2])) + "1")

event_table = pa.Table.from_pandas(events_df)

write_path = os.path.join("/home/svjack/temp_dir" ,"event_log")
### write it to local
pq.write_to_dataset(event_table, write_path, partition_cols=["event", "time_two_pos"])

### read by condition
filters_objects = [[("event", "in", ["transaction", "addtocart"]), ("time_two_pos", ">", pd.to_datetime("1970-01-01 00:24:01.200000001"))]]
filters_types = [[("event", "str"), ("time_two_pos", "timestamp[ms]")]]

p_path = write_path

Timestamp = pd.Timestamp
exp_cast_ds = _ParquetDatasetV2PartitionCastWellFormatted(
        p_path,
        filters_objects = filters_objects,
        filters_types = filters_types)

all_filtered_df_in_upper = exp_cast_ds.read().to_pandas()

assert np.all(pd.to_datetime(all_filtered_df_in_upper["time_two_pos"]) > pd.to_datetime("1970-01-01 00:24:01.200000001"))

filters_objects = [[("event", "in", ["transaction", "addtocart"]), ("time_two_pos", "<=", pd.to_datetime("1970-01-01 00:24:01.200000001"))]]
filters_types = [[("event", "str"), ("time_two_pos", "timestamp[ms]")]]

Timestamp = pd.Timestamp
exp_cast_ds = _ParquetDatasetV2PartitionCastWellFormatted(
        p_path,
        filters_objects = filters_objects,
        filters_types = filters_types)

all_filtered_df_in_lower = exp_cast_ds.read().to_pandas()

assert np.all(pd.to_datetime(all_filtered_df_in_lower["time_two_pos"]) <= pd.to_datetime("1970-01-01 00:24:01.200000001"))

filters_objects = [[("event", "in", ["transaction", "addtocart"]),]]
filters_types = [[("event", "str"),]]


exp_cast_ds = _ParquetDatasetV2PartitionCastWellFormatted(
        p_path,
        filters_objects = filters_objects,
        filters_types = filters_types)

all_filtered_df_in = exp_cast_ds.read().to_pandas()

assert all_filtered_df_in.shape[0] == all_filtered_df_in_lower.shape[0] + all_filtered_df_in_upper.shape[0]


