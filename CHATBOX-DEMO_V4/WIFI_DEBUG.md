# Jetson WiFi Hotspot Recovery

Fixing a NetworkManager AP on a Jetson that broadcasts fine but rejects every client — including the case where `nmcli connection modify` silently fails to reload the key.

## Symptoms

- ESP32 reports `Disconnect reason: 15` (`WIFI_REASON_4WAY_HANDSHAKE_TIMEOUT`)
- Phone shows "incorrect password" / "authentication error" despite the password being correct
- `nmcli` reads back the right PSK from the profile, but joining still fails

All three mean the same thing: the client found the AP and the SSID matched, but the WPA handshake failed. The running AP is enforcing a key that doesn't match the one in the profile.

## Requirements

- NetworkManager (`nmcli`)
- `iw`
- A WiFi interface capable of AP mode
- A second network path (ethernet, or a second WiFi interface) if you're working over SSH

## 1. Find the profile that's actually live

Do this first. It is easy to spend an hour editing a profile that isn't running.

```bash
nmcli connection show --active
```

Find the row whose `DEVICE` is your AP interface (e.g. `wlan0`). The `NAME` column is the profile to work with — it is **not** necessarily the one you think.

Confirm the radio is genuinely in AP mode:

```bash
iw dev
```

The interface should report `type AP` and the expected `ssid`. If it says `type managed`, the hotspot isn't up at all and this guide doesn't apply.

## 2. Check credentials on that profile

```bash
sudo nmcli -s -g 802-11-wireless-security.psk connection show <PROFILE>
nmcli -f 802-11-wireless-security.key-mgmt,802-11-wireless.band connection show <PROFILE>
```

| Field | Expected | Why |
|---|---|---|
| `psk` | your password | mismatch causes reason 15 |
| `key-mgmt` | `wpa-psk` | `sae` (WPA3) is not supported by ESP32 |
| `band` | `bg` | ESP32 is 2.4 GHz only and cannot see 5 GHz |

If the PSK is simply wrong, update and restart — sometimes this is enough:

```bash
sudo nmcli connection modify <PROFILE> 802-11-wireless-security.psk "yourpassword"
sudo nmcli connection down <PROFILE>
sudo nmcli connection up <PROFILE>
```

Test with a phone. If it joins, stop here.

## 3. Delete and recreate

If the config reads back correctly but clients still can't join, the modify didn't take effect on the running AP. Recreate the profile from scratch:

```bash
sudo nmcli connection delete ChatBox-AP
sudo nmcli device wifi hotspot ifname wlan0 ssid ChatBox-AP password chatbox1234
sudo nmcli connection modify Hotspot 802-11-wireless.band bg
sudo nmcli connection up Hotspot
```

Substitute your own interface, SSID, and password.

## 4. Rename the profile

`nmcli device wifi hotspot` names the new profile `Hotspot` regardless of the SSID. To keep the profile name aligned with the SSID:

```bash
sudo nmcli connection modify Hotspot connection.id ChatBox-AP
sudo nmcli connection up ChatBox-AP
```

> The profile name and the broadcast SSID are separate. The SSID was already `ChatBox-AP`; this only makes future `nmcli` commands consistent.

## 5. Verify

```bash
nmcli connection show --active
sudo nmcli -s -g 802-11-wireless-security.psk connection show ChatBox-AP
```

**Join from a phone before flashing the ESP32.** One phone test separates AP problems from firmware problems and saves a lot of guessing.

Once a client is connected:

```bash
ip neigh show dev wlan0
```

## Gotchas

**Delete stale profiles.** Leftover profiles serving the same purpose under different names (an old `Hotspot` alongside `ChatBox-AP`) are the root cause of this whole class of bug — you check one, the other is running.

```bash
sudo nmcli connection delete <stale-name>
```

**Don't touch the interface carrying your SSH session.** Bringing the AP down/up is safe if you're connected over a different interface. If you're SSH'd in through the hotspot itself, you will lock yourself out — use a monitor and keyboard.

**Band must be `bg`.** ESP32 cannot see 5 GHz networks at all.

**Watch the handshake live** if it still fails. Run this in a second terminal while a client attempts to join:

```bash
sudo journalctl -u NetworkManager -f
```

## ESP32 client sketch

Minimal sketch that prints the disconnect reason code, useful for diagnosing:

```cpp
#include <WiFi.h>

void WiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.print("Disconnect reason: ");
    Serial.println(info.wifi_sta_disconnected.reason);
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.onEvent(WiFiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true, true);   // clear cached credentials
  delay(1000);
  WiFi.begin("ChatBox-AP", "chatbox1234");
}

void loop() {
  Serial.println(WiFi.status() == WL_CONNECTED
                 ? WiFi.localIP().toString() : "not connected");
  delay(2000);
}
```

| Reason | Meaning | Likely cause |
|---|---|---|
| 15 | 4-way handshake timeout | wrong PSK, or WPA3-only AP |
| 201 | no AP found | wrong SSID, or AP on 5 GHz |
| 202 | auth fail | wrong password |
| 205 | connection lost | weak signal, AP restarted |
