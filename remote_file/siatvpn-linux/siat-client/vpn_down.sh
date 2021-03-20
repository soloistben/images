#!/bin/sh

TAP=$1

ps -ef | grep "dhclient $TAP" | awk '{print $2}' | xargs kill -9 2> /dev/null

echo ""
echo " ------  DONE ! ------"
echo ""
sleep 1
