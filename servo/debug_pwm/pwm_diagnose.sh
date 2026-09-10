#!/bin/bash

echo "============================================================"
echo " JETSON ORIN NANO PWM DIAGNOSTIC"
echo "============================================================"
echo

echo "===== OS ====="
cat /etc/nv_tegra_release 2>/dev/null
uname -a
echo

echo "===== PWM CHIPS ====="

for CHIP in /sys/class/pwm/pwmchip*; do
    echo
    echo "------------------------------------------------------------"
    echo "CHIP: $CHIP"
    echo "------------------------------------------------------------"

    echo "Device:"
    readlink -f "$CHIP/device"

    echo "npwm:"
    cat "$CHIP/npwm"

    echo "uevent:"
    cat "$CHIP/device/uevent" 2>/dev/null

    echo
    echo "PWM channels:"

    for PWM in "$CHIP"/pwm*; do
        [ -d "$PWM" ] || continue

        echo
        echo "  $PWM"

        echo "    period:"
        cat "$PWM/period" 2>/dev/null

        echo "    duty:"
        cat "$PWM/duty_cycle" 2>/dev/null

        echo "    enabled:"
        cat "$PWM/enable" 2>/dev/null

        echo "    polarity:"
        cat "$PWM/polarity" 2>/dev/null
    done
done

echo
echo "===== 40-PIN PWM INFORMATION ====="

echo
echo "Jetson GPIO/PWM related information:"
grep -R "PWM" /sys/kernel/debug/gpio 2>/dev/null | head -50

echo
echo "===== PINMUX / JETSON-IO ====="

if [ -x /opt/nvidia/jetson-io/jetson-io.py ]; then
    echo "Jetson-IO exists:"
    ls -l /opt/nvidia/jetson-io/jetson-io.py
else
    echo "Jetson-IO not found."
fi

echo
echo "===== CLOCK INFORMATION ====="

if [ -f /sys/kernel/debug/clk/clk_summary ]; then
    grep -i pwm /sys/kernel/debug/clk/clk_summary
else
    echo "clk_summary unavailable"
fi

echo
echo "===== PWM DEVICE TREE LINKS ====="

for CHIP in /sys/class/pwm/pwmchip*; do
    echo
    echo "$CHIP -> $(readlink -f "$CHIP/device")"
done

echo
echo "============================================================"
echo " END"
echo "============================================================"
