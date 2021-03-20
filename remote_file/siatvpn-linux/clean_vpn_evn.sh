#!/bin/sh

TAP=tap0

ps -ef | grep "dhclient $TAP" | awk '{print $2}' | xargs kill -9

ps -ef | grep "vpn_up" | awk '{print $2}' | xargs kill -9
ps -ef | grep "ping -c 3" | awk '{print $2}' | xargs kill -9
ps -ef | grep "openvpn" | awk '{print $2}' | xargs kill -9


