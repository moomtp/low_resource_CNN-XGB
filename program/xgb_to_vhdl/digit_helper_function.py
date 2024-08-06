import numpy as np
import warnings
import struct

def f_to_b(val:float):
    """float_to_float16_binary"""
    if not(-65515 < val < 65515):
        warnings.warn("overflow occur in node's val")
    # bytes_representation = np.float16(val).tobytes() # binary format in little endian
    bytes_representation = np.float16(val).newbyteorder('big').tobytes() # binary format in big endian
    binary_format = ''.join(f'{byte:08b}' for byte in bytes_representation)[:16] 
    # print(binary_format)
    return binary_format


def b_to_f(b_str):
    """binary string to float16"""
    # Ensure the binary string is 16 bits long
    if len(b_str) != 16:
        raise ValueError("Binary string must be exactly 16 bits long")
    
    # Convert binary string to an integer
    int_value = int(b_str, 2)
    
    # Pack integer as bytes in little-endian format
    packed = struct.pack('<H', int_value)
    
    # Use numpy to interpret the bytes as float16 and then convert to float64 for Python usability
    float_value = np.frombuffer(packed, dtype=np.float16)[0].astype(float)
    
    return float_value


def i_to_b(val:int, length) -> str: 
    """int_to_n's_binary"""
    binary_string = bin(val)[2:]  # 5 -> 0b0101 -> 0101
    return binary_string.zfill(length)  


def f_to_fix16_b(value, integer_bits=8, fractional_bits=7):
    # default is Q8.7
    # 計算放大倍數
    multiplier = 2 ** fractional_bits
    
    # 轉換到固定點數
    fixed_point_value = int(round(value * multiplier))
    
    # 範圍檢查
    min_val = -(2 ** (integer_bits + fractional_bits - 1))
    max_val = (2 ** (integer_bits + fractional_bits - 1)) - 1
    
    # 調整範圍使其適合16位有符號整數
    fixed_point_value = max(min_val, min(max_val, fixed_point_value))
    
    # 轉換為16位二進位字符串，並返回
    return format(fixed_point_value & 0xFFFF, '016b')

def b_to_i(str):
    # binary to int
    return int("0b" + str , 2)

def b_to_h(binary_string:str) -> str: 
    """binary_to_hex"""
    integer_value = int(binary_string, 2)
    hex_value = format(integer_value, '0>8x')
    # hex_value = hex(integer_value)
    # print(hex_value)
    return hex_value

# def h_to_b(self, binary_string:str) -> str: 
#     return 0

