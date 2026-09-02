import json

class RoleRegistry:
    def __init__(self, domain, role):
        self.domain = domain
        self.role = role
        self.role_profile = self.get_role_profile()
    
    def get_role_profile(self):
        # --- shared-layer shim (agent_wf_v2) --- dataset role profile v1
        # The agent's profile is read from disk here, by task-type name, so a
        # dataset-specific pool has to be preferred at THIS point. Patching
        # encoder_roles alone only changed which role the router picks, not the
        # text that role is then given: measured in the live transcripts, 15.6% of
        # DROP prompts still said "choose the correct answer" and the adapted
        # wording appeared zero times. See shims/masrouter/install.py.
        import os as _shim_rr_os

        _shim_rr_ds = _shim_rr_os.getenv("SHIM_DATASET", "")
        if _shim_rr_ds:
            _shim_rr_path = f"MAR/Roles/{self.domain}_{_shim_rr_ds}/{self.role}.json"
            if _shim_rr_os.path.exists(_shim_rr_path):
                return json.load(open(_shim_rr_path, encoding="utf-8"))
        profile = json.load(open(f"MAR/Roles/{self.domain}/{self.role}.json"))
        return profile
    
    def get_name(self):
        return self.role_profile['Name']
    
    def get_message_aggregation(self):
        return self.role_profile['MessageAggregation']
    
    def get_description(self):
        return self.role_profile['Description']
    
    def get_output_format(self):
        return self.role_profile['OutputFormat']
    
    def get_reasoning(self):
        return self.role_profile['Reasoning']
    
    def get_post_process(self):
        return self.role_profile['PostProcess']
    
    def get_post_description(self):
        return self.role_profile['PostDescription']
    
    def get_post_output_format(self):
        return self.role_profile['PostOutputFormat']
    