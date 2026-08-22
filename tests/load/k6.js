import http from "k6/http";
import {check, sleep} from "k6";

export const options = {
  scenarios: {steady: {executor: "ramping-vus", stages: [
    {duration: "30s", target: 10}, {duration: "2m", target: 10}, {duration: "30s", target: 0},
  ]}},
  thresholds: {http_req_failed: ["rate<0.01"], http_req_duration: ["p(95)<2000"]},
};

const base = __ENV.BASE_URL || "http://127.0.0.1:8000";
export default function () {
  const response = http.get(`${base}/api/health`);
  check(response, {"health is 200": value => value.status === 200});
  sleep(1);
}
