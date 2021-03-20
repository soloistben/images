#!/bin/sh

cd siatvpn-linux/siat-client

set timeout 1
spawn openvpn --config ./siat-client.ovpn
expect "Enter Auth Username:"
send "XJY003489\r"
expect "Enter Auth Password:"
send "pwd156073551\r"
interact

