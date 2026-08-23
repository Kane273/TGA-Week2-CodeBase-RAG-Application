import http from "k6/http";
import {check, sleep} from "k6";

export const options = {
  scenarios: {
    health: {executor: "constant-arrival-rate", exec: "health", rate: 20,
      timeUnit: "1s", duration: "2m", preAllocatedVUs: 10},
    workspace_reads: {executor: "ramping-vus", exec: "workspaces", startVUs: 0,
      stages: [{duration: "30s", target: 10}, {duration: "2m", target: 25},
        {duration: "30s", target: 0}]},
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{endpoint:health}": ["p(95)<500"],
    "http_req_duration{endpoint:workspaces}": ["p(95)<2000"],
  },
};

const base = __ENV.BASE_URL || "http://127.0.0.1:8000";
export function health() {
  const response = http.get(`${base}/api/health`, {tags: {endpoint: "health"}});
  check(response, {"health is 200": value => value.status === 200});
}

export function workspaces() {
  const response = http.get(`${base}/api/local-sources`, {tags: {endpoint: "workspaces"}});
  check(response, {"workspace list is 200": value => value.status === 200});
  sleep(1);
}
