/*
* Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE
*/

#include <stdlib.h>
#include <stdio.h>
#include "gpuvsmi.h"

#ifdef WINDOW_MODE
#include <ncurses.h>
#else
#define printw(...) printf(__VA_ARGS__)
#endif

#define CHECKRET(a) do { \
	do { \
		ret = a; \
		if (ret != -SMI_ERR_RETRY) \
			break; \
		usleep(500*1000); \
	} while (true); \
	if (ret != SMI_SUCCESS) { \
		printw("GPU%d Error in line %d : %s failed with ret=%d\n", \
				i, __LINE__, #a, ret); \
		goto fini; \
	} \
} while (0)

static const char* get_driver_name(uint32_t id)
{
	switch(id) {
	case SMI_DRIVER_LIBGV:
		return "GIM.L";
	case SMI_DRIVER_KMD:
		return "AMD-KMD";
	case SMI_DRIVER_AMDGPUV:
		return "MSGPUV";
	case SMI_DRIVER_AMDGPU:
		return "AMDGPU";
	default:
		break;
	}
	return "Unknown";
}

static const char * __str_sched_state(enum smi_vf_sched_state status)
{
	switch (status) {
	case VF_STATE_UNAVAILABLE:
		return "UNAVL";
	case VF_STATE_AVAILABLE:
		return "AVAIL";
	case VF_STATE_ACTIVE:
		return "ACTIV";
	case VF_STATE_SUSPENDED:
		return "SUSPN";
	case VF_STATE_FULLACCESS:
		return "FACCS";
        case VF_STATE_DEFAULT_AVAILABLE:
		return "DEFAV";
	default:
		break;
	}

	return "UNKNW";
}

static const char * __drv_err(enum smi_driver_status status)
{
	switch (status) {
		case DRV_STATUS_SUCCESS:
			return "driver load success";
		case DRV_STATUS_HW_FAIL:
			return "gpu failed initialization";
		case DRV_STATUS_HW_LOST:
			return "gpu hw failed during runtime";
		case DRV_STATUS_LOAD_FAIL:
			return "driver failed to init sw structures";
		default:
			break;
	}
	return "";
}

float convert_volt(int v)
{
	return v / 1000.0;
}

int main()
{
	int ret = 0;
	unsigned int gpu_count;
	uint8_t vf_num;
	struct smi_device_handle devices[SMI_MAX_DEVICES];
	struct smi_asic_info asic;
	struct smi_bus_info bus;
	struct smi_power_info power;
	struct smi_gpu_caps caps;
	struct smi_pf_fb_info fb;
	struct smi_vbios_info vbios;
	struct smi_ucode_info ucode;
	struct smi_vf_config config;
	struct smi_partition_info partitions[SMI_MAX_VF_COUNT];
	struct smi_host_driver_info host_info;
	struct smi_engine_usage usage;
	struct smi_power_measure measure;
	struct smi_gpu_clock_measure clock;
	struct smi_gpu_thermal_measure temp_edge, temp_hotspot, temp_mem;
	struct smi_sched_info sched_info;
	struct smi_board_info board_info;
	union smi_bdf bdf;
	int i, j, k;

	uint64_t uce, cce;

	ret = gpuvsmi_init_flags(false);
	if (ret != SMI_SUCCESS)
		return ret;

	gpu_count = SMI_MAX_DEVICES;
	ret = gpuvsmi_get_devices(&gpu_count, devices);
	if (ret != SMI_SUCCESS)
		goto fini;

#ifdef WINDOW_MODE
	initscr();
	nodelay(stdscr, true);

	while (true) {
	clear();
	refresh();			/* Print it on to the real screen */
#endif

	for (i=0; i < gpu_count; i++) {
		CHECKRET(gpuvsmi_get_bus_info(devices[i], &bus));
		CHECKRET(gpuvsmi_get_asic_info(devices[i], &asic));
		ret = gpuvsmi_get_host_driver_version(devices[i], &host_info);
		if ((ret != SMI_SUCCESS) || (DRV_STATUS_SUCCESS != host_info.status))
		{
			printw("GPU[%08lx] BDF:[%02x:%02x.%d] [%04x:%04x] - %s \n",
					devices[i].handle,
					bus.pcie.bdf.bus_number,
					bus.pcie.bdf.device_number,
					bus.pcie.bdf.function_number,
					asic.vendor_id,
					asic.device_id,
					__drv_err(host_info.status));
			continue;
		}

                if (bus.pcie.bdf.bus_number != 0x22) continue ;

		ret = gpuvsmi_get_num_vf_enabled(devices[i], &vf_num);
		if (ret != SMI_SUCCESS)
			break;
	
		CHECKRET(gpuvsmi_get_board_info(devices[i], &board_info));
		CHECKRET(gpuvsmi_get_power_info(devices[i], &power));
		CHECKRET(gpuvsmi_get_caps_info(devices[i], &caps));
		CHECKRET(gpuvsmi_get_fb_info(devices[i], &fb));
		CHECKRET(gpuvsmi_get_vbios_info(devices[i], &vbios));
		CHECKRET(gpuvsmi_get_ucode_info(devices[i], &ucode));
		CHECKRET(gpuvsmi_get_gpu_activity(devices[i], &usage));
		CHECKRET(gpuvsmi_get_power_measure(devices[i], &measure));
		CHECKRET(gpuvsmi_get_thermal_measure(devices[i],
			THERMAL_DOMAIN_EDGE, &temp_edge));
		CHECKRET(gpuvsmi_get_thermal_measure(devices[i],
			THERMAL_DOMAIN_MEM, &temp_mem));
		CHECKRET(gpuvsmi_get_thermal_measure(devices[i],
			THERMAL_DOMAIN_HOTSPOT, &temp_hotspot));

		CHECKRET(gpuvsmi_get_ecc_error_count(devices[i], &uce, &cce));

		printw("GPU[%08lx] BDF:[%02x:%02x.%d] [%04x:%04x] %s %dMB "
			"GFX%d.%d %dcu TDP: %dW VF_MAX: %d ECC: %s\n",
						devices[i].handle,
						bus.pcie.bdf.bus_number,
						bus.pcie.bdf.device_number,
						bus.pcie.bdf.function_number,
						asic.vendor_id,
						asic.device_id,
						asic.market_name,
						fb.total_fb_size,
						caps.gfx.gfxip_major,
						caps.gfx.gfxip_minor,
						caps.gfx.gfxip_cu_count,
						power.power_cap,
						caps.max_vf_num,
						caps.ecc_supported ? "ON" : "OFF"
						);
		printw(" S/N: %016lx", asic.asic_serial);
		printw(" [%s] Model: %s Serial: %s FRU_ID: %s\n",
			(board_info.is_master) ? "master" : "slave",
			board_info.model_number,
			board_info.product_serial,
			board_info.fru_id);
#if 1
		for (k = 0; k < host_info.num_driver; k++)
			printw("-->Driver[%d] - %s v:%x build date:%s\n",
					k,
					get_driver_name(host_info.drivers[k].id),
					host_info.drivers[k].version,
					host_info.drivers[k].build_date);
#endif

		printw("+VBIOS ver:%x %s %s\n",
					vbios.vbios_version, vbios.build_date,
					vbios.part_number);
		CHECKRET(gpuvsmi_get_clock_measure(devices[i],
					CLOCK_DOMAIN_GFX, &clock));
		printw("+GFX %d%% avg %dMhz cur: %dMhz\n",
			usage.gfx_usage, clock.avg_clk, clock.cur_clk);
		CHECKRET(gpuvsmi_get_clock_measure(devices[i],
					CLOCK_DOMAIN_MEM, &clock));

		if (caps.supported_fields_flags & MEM_USAGE_FLAG) {
			printw("+MEM %d%% avg %dMhz cur: %dMhz\n",
				usage.mem_usage,
				clock.avg_clk, clock.cur_clk);
		}
		else {
			printw("+MEM --N/A-- avg %dMhz cur: %dMhz\n",
				clock.avg_clk, clock.cur_clk);
		}
		CHECKRET(gpuvsmi_get_clock_measure(devices[i],
					CLOCK_DOMAIN_MM, &clock));
		if (caps.supported_fields_flags & MM_METRICS_FLAG) {
			printw("+ENC %d%% avg %dMhz cur: %dMhz\n",
				usage.mm_usage,
				clock.avg_clk, clock.cur_clk);
		}
		else {
			printw("+ENC --N/A-- avg --N/A-- cur: --N/A--\n");
		}
		if (caps.supported_fields_flags & POWER_GFX_VOLTAGE_FLAG) {
			printw("+Power: %dW (%ldmJ) Voltage: %.4fV Edge: %dC Hotspot: %dC Mem: %dC\n",
						measure.power,
						measure.energy,
						convert_volt(measure.voltage),
						temp_edge.temperature,
						temp_hotspot.temperature,
						temp_mem.temperature);
		}
		else {
			printw("+Power: %dW (%ldmJ) Voltage: --N/A-- Edge: %dC Hotspot: %dC Mem: %dC\n",
						measure.power,
						measure.energy,
						temp_edge.temperature,
						temp_hotspot.temperature,
						temp_mem.temperature);
		}

		printw("+ECC: UCE: %ld CCE: %ld\n", uce, cce);

		ret = gpuvsmi_get_vf_partition_info(devices[i],
						vf_num, partitions);
		if (ret != SMI_SUCCESS)
			break;
		printw("+ Num VF Enabled: %d\n", vf_num);

		for (j = 0; j < vf_num ; j++) {
			CHECKRET(gpuvsmi_get_device_bdf(partitions[j].id,
						&bdf));
			CHECKRET(gpuvsmi_get_vf_config(partitions[j].id,
						&config));
			CHECKRET(gpuvsmi_get_sched_info(partitions[j].id,
						&sched_info));
			printw("  -- vf[%08lx] %02x:%02x.%d flags: %08lx "
				"\tFB size: %dMB offset: %dMB GFX ts:%dusec sched[%s]\n",
				partitions[j].id.handle,
				bdf.bus_number,
				bdf.device_number,
				bdf.function_number,
				config.flags,
				config.fb.fb_size,
				config.fb.fb_offset,
				config.gfx_timeslice_us,
				__str_sched_state(sched_info.state));
		}
	}

#ifdef WINDOW_MODE
	/* Wait for user input or execution error */
	if (getch() == 'q' || ret != SMI_SUCCESS)
		break;
	usleep(500*1000);
	} /* while (true) */

	endwin();
#endif

fini:
	ret = gpuvsmi_fini();
	if (ret != SMI_SUCCESS)
		printf("GPUV SMI failed to finish\n");

	return ret;
}
