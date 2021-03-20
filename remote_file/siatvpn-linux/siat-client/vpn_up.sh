#!/bin/bash

#######################################################
# a scripts to init vpn client connect.
#      RT Luo . 201912
#      MXNetwork Limited
#######################################################

source /etc/profile

TAP=$1

ip link set $TAP up
sleep 1

setuprt()
{

ip address add 0.0.0.0/0 dev $TAP
sleep 1
dhclient $TAP

G=1
K=1

while ([ $G == 1 ] && [ $K -le 10 ])
do
  ping -c 3  172.20.48.1
  AG=$?
  if [ $AG == 0 ]
     then
        G=0
  fi
  K=`expr $K + 1`
  if [ $K > 10 ]
    then
         echo " "
         echo " ------------- SIAT VPN not Connected ! ! ! -------------"
         echo " "
  fi
done

add_route()
{  
    #/usr/sbin/ip -6 route add default  via 2001:250:3c02:719::1
	ip route add 172.20.0.0/16 via 172.20.48.1
	ip route add default via 172.20.48.1
    BG=$?
    if [ $BG == 0 ]
       then 
         echo " "
         echo " ------------- SIAT VPN Connected ! ! ! --------------- "
         echo " "
       else
         echo " "
         echo " ------------- SIAT VPN not Connected ! ! ! -------------"
         echo " "
    fi
}

if [ $AG == 0 ]
   then
     add_route 
fi

}

setuprt &

