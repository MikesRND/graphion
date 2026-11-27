"""A stand-in for one browser window, run as its own process by bench.py.

    python -m spikes.inspection.bench_client HOST PORT HZ PATH REQUEST_JSON

It posts REQUEST_JSON to PATH over one kept-alive connection, HZ times a
second (as fast as it can when HZ is 0), and follows the server's ``since``
protocol when the answer is a binary observation. It prints ``ready`` after
the first answer, and one JSON line of totals when its standard input closes.
"""

import http.client
import json
import struct
import sys
import threading
import time


def main():
    """Poll until standard input closes, then report what was received."""
    host, port, hz, path, request = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
    request = json.loads(request)
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    connection = http.client.HTTPConnection(host, port)
    polls = updates = received = 0
    since = None
    ready = False
    started = due = time.perf_counter()
    while not stop.is_set():
        body = json.dumps({**request, "since": since} if path.endswith("observe") else request)
        connection.request("POST", path, body)
        response = connection.getresponse()
        data = response.read()
        polls += 1
        if response.status == 200:
            updates += 1
            received += len(data)
            if response.getheader("Content-Type") != "application/json":
                (length,) = struct.unpack_from("<I", data)
                since = json.loads(data[4 : 4 + length])["serial"]
        if not ready:
            # The first answer is connection set-up and a full snapshot; count from the second.
            ready = True
            print("ready", flush=True)
            started = due = time.perf_counter()
            polls = updates = received = 0
        if hz:
            due += 1.0 / hz
            time.sleep(max(0.0, due - time.perf_counter()))
    seconds = time.perf_counter() - started
    print(json.dumps({"polls": polls, "updates": updates, "bytes": received, "seconds": seconds}), flush=True)


if __name__ == "__main__":
    main()
