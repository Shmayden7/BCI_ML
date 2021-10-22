import socket
import struct

class liveEEG:

    def getData(UDP_IP, UDP_PORT):
        # hostname = socket.gethostname()
        # UDP_IP = socket.gethostbyname(hostname)
            
        sock = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        sock.bind((UDP_IP, UDP_PORT))
        

        value = ''
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        print(data)
        for i in range(len(data) - 4,len(data)):   
            value += str(hex(data[i]).split('x')[-1])
            if (len(value) % 2 != 0):
                value = value[:len(value)-1] +'0'+ value[len(value)-1:]
        real_value = struct.unpack('!f', bytes.fromhex(value))[0]
        
        return real_value


